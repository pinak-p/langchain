import json
import multiprocessing as mp
import os
from functools import partial
from typing import Any, Dict, List, Optional

import boto3
from pydantic import BaseModel, Extra, root_validator

from langchain.embeddings.base import Embeddings


class BedrockEmbeddings(BaseModel, Embeddings):
    """Bedrock embedding models.

    To authenticate, the AWS client uses the following methods to
    automatically load credentials:
    https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html

    If a specific credential profile should be used, you must pass
    the name of the profile from the ~/.aws/credentials file that is to be used.

    Make sure the credentials / roles used have the required policies to
    access the Bedrock service.
    """

    """
    Example:
        .. code-block:: python

            from langchain.bedrock_embeddings import BedrockEmbeddings
            
            region_name ="us-east-1"
            credentials_profile_name = "default"
            model_id = "amazon.titan-e1t-medium"

            be = BedrockEmbeddings(
                credentials_profile_name=credentials_profile_name,
                region_name=region_name,
                model_id=model_id
            )
    """

    client: Any  #: :meta private:
    """Bedrock client."""
    region_name: Optional[str] = None
    """The aws region e.g., `us-west-2`. Fallsback to AWS_DEFAULT_REGION env variable
    or region specified in ~/.aws/config in case it is not provided here.
    """

    credentials_profile_name: Optional[str] = None
    """The name of the profile in the ~/.aws/credentials or ~/.aws/config files, which
    has either access keys or role information specified.
    If not specified, the default credential profile or, if on an EC2 instance,
    credentials from IMDS will be used.
    See: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html
    """

    model_id: str = "amazon.titan-e1t-medium"
    """Id of the model to call, e.g., amazon.titan-e1t-medium, this is
    equivalent to the modelId property in the list-foundation-models api"""

    model_kwargs: Optional[Dict] = None
    """Key word arguments to pass to the model."""

    endpoint_url: Optional[str] = None
    """Needed if you don't want to default to us-east-1 endpoint"""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that AWS credentials to and python package exists in environment."""

        if values["client"] is not None:
            values["region_name"] = values["client"].meta.region_name
            values["endpoint_url"] = values["client"].meta.endpoint_url
            return values

        try:
            if values["credentials_profile_name"] is not None:
                session = boto3.Session(profile_name=values["credentials_profile_name"])
            else:
                # use default credentials
                session = boto3.Session()

            client_params = {}
            if values["region_name"]:
                client_params["region_name"] = values["region_name"]

            if values["endpoint_url"]:
                client_params["endpoint_url"] = values["endpoint_url"]

            values["client"] = session.client("bedrock", **client_params)
            values["region_name"] = values["client"].meta.region_name
            values["endpoint_url"] = values["client"].meta.endpoint_url

        except ImportError:
            raise ModuleNotFoundError(
                "Could not import boto3 python package. "
                "Please install it with `pip install boto3`."
            )
        except Exception as e:
            raise ValueError(
                "Could not load credentials to authenticate with AWS client. "
                "Please check that credentials in the specified "
                "profile name are valid."
            ) from e

        return values

    def _embedding_func(self, text: str) -> List[float]:
        """Call out to Bedrock embedding endpoint."""
        # replace newlines, which can negatively affect performance.
        text = text.replace(os.linesep, " ")
        _model_kwargs = self.model_kwargs or {}

        input_body = {**_model_kwargs, "inputText": text}
        body = json.dumps(input_body)

        try:
            response = self.client.invoke_model(
                body=body,
                modelId=self.model_id,
                accept="application/json",
                contentType="application/json",
            )
            response_body = json.loads(response.get("body").read())
            return response_body.get("embedding")
        except Exception as e:
            raise ValueError(f"Error raised by inference endpoint: {e}")

    def embed_documents(
        self, texts: List[str], chunk_size: int = 1
    ) -> List[List[float]]:
        """Compute doc embeddings using a Bedrock model.

        Args:
            texts: The list of texts to embed.
            chunk_size: Bedrock currently only allows single string
                inputs, so chunk size is always 1. This input is here
                only for compatibility with the embeddings interface.


        Returns:
            List of embeddings, one for each text.
        """
        results = []
        for text in texts:
            response = self._embedding_func(text)
            results.append(response)
        return results

    def _embedding_multiprocessing_func(self, text: str) -> List[float]:
        """Call out to Bedrock embeddings endpoint. This has the same functionality as
        _embedding_func() except that it creates a boto3 bedrock client rather than
        using the class's member variable (this is needed for multiprocessing because the
        boto3 session objects are not thread safe - https://boto3.amazonaws.com/v1/documentation/api/latest/guide/session.html#multithreading-or-multiprocessing-with-sessions).
        """

        # replace newlines, which can negatively affect performance.
        text = text.replace(os.linesep, " ")
        model_kwargs = self.model_kwargs or {}

        input_body = {**model_kwargs, "inputText": text}
        body = json.dumps(input_body)

        client_kwargs = {}

        if self.region_name:
            client_kwargs["region_name"] = self.region_name

        if self.endpoint_url:
            client_kwargs["endpoint_url"] = self.endpoint_url

        bedrock = boto3.client(service_name="bedrock", **client_kwargs)
        try:
            response = bedrock.invoke_model(
                body=body,
                modelId=self.model_id,
                accept="application/json",
                contentType="application/json",
            )
            response_body = json.loads(response.get("body").read())

            embedding = response_body.get("embedding")
        except Exception as e:
            if e.response["Error"]["Code"] == "ThrottlingException":
                # sleep before retry
                time.sleep(0.1)

                try:
                    response = bedrock.invoke_model(
                        body=body,
                        modelId=self.model_id,
                        accept="application/json",
                        contentType="application/json",
                    )
                    response_body = json.loads(response.get("body").read())
                    embedding = response_body.get("embedding")
                except Exception as e:
                    raise ValueError(f"Error raised by inference endpoint: {e}")
            else:
                raise ValueError(f"Error raised by inference endpoint: {e}")

        return embedding

    def embed_documents_multiprocessing(
        self, texts: List[str], num_jobs: int = mp.cpu_count()
    ) -> List[List[float]]:
        """
        Compute embeddings for texts using multiprocessing.
        i.e. parallel invocations to Bedrock API.

        Args:
            texts: The list of texts to embed.
            num_jobs: Number of parallel processes to spawn. Defaults to cpu count.
        Returns:
            List of embeddings, one for each text.

        """

        # The client object cannot be pickled (needed for multiprocessing to work),
        # save the client object into a temporary variable and restore it later.
        # A new client object is now created inside each process. This class is not
        # re-entrant so this is ok.
        client = self.client
        self.client = None

        # create a process pool for the number of jobs to run in parallel
        pool = mp.Pool(num_jobs)

        # distribute the input list between the processes in the pool (the use of pool.imap
        # returns the output (embeddings) in the same order as the input texts)
        embeddings = list(
            pool.imap(partial(self._embedding_multiprocessing_func), texts)
        )

        # restore the client back
        self.client = client

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Compute query embeddings using a Bedrock model.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self._embedding_func(text)
