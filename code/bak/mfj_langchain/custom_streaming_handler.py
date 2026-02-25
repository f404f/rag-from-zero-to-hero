import sys

from langchain_core.callbacks import BaseCallbackHandler


class CustomStreamingHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token, **kwargs):
        sys.stdout.write(token)
        sys.stdout.flush()
