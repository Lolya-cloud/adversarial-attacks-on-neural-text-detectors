class Parameters:
    """
    max_tokens: integer, Optional, Defaults to 16. Maximum number of tokens to generate in the completion.

    temperature: number, Optional, Defaults to 1. Sampling temperature between 0 and 2.

    top_p: number, Optional, Defaults to 1. Nucleus sampling parameter where model considers tokens with top_p probability mass.

    n: integer, Optional, Defaults to 1. Number of completions to generate for each prompt.

    stream: boolean, Optional, Defaults to false. Whether to stream back partial progress.

    logprobs: integer, Optional, Defaults to null. Include log probabilities on the logprobs most likely tokens.

    echo: boolean, Optional, Defaults to false. Echo back the prompt in addition to the completion.

    stop: string or array, Optional, Defaults to null. Up to 4 sequences where the API will stop generating further tokens.

    presence_penalty: number, Optional, Defaults to 0. Number between -2.0 and 2.0. Penalizes new tokens based on whether they appear in the text so far.

    frequency_penalty: number, Optional, Defaults to 0. Number between -2.0 and 2.0. Penalizes new tokens based on their existing frequency in the text so far.

    best_of: integer, Optional, Defaults to 1. Generates best_of completions server-side and returns the "best".

    logit_bias: map, Optional, Defaults to null. Modify the likelihood of specified tokens appearing in the completion.
    """
    def __init__(self, max_tokens=16, temperature=1, top_p=1, n=1, stream=False, logprobs=None,
                 echo=False, stop=None, presence_penalty=0, frequency_penalty=0, best_of=1, logit_bias=None):

        if not 0 <= max_tokens <= 4096:
            raise ValueError("max_tokens must be between 0 and 4096")
        if not 0 <= temperature <= 2:
            raise ValueError("temperature must be between 0 and 2")
        if not 0 <= top_p <= 1:
            raise ValueError("top_p must be between 0 and 1")
        if n < 1:
            raise ValueError("n must be greater than 0")
        if logprobs is not None and (logprobs < 0 or logprobs > 5):
            raise ValueError("logprobs must be between 0 and 5")
        if not -2 <= presence_penalty <= 2:
            raise ValueError("presence_penalty must be between -2 and 2")
        if not -2 <= frequency_penalty <= 2:
            raise ValueError("frequency_penalty must be between -2 and 2")
        if best_of < 1:
            raise ValueError("best_of must be greater than 0")

        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.n = n
        self.stream = stream
        self.logprobs = logprobs
        self.echo = echo
        self.stop = stop
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.best_of = best_of
        self.logit_bias = logit_bias

    def to_dict(self):
        return {k: v for k, v in {
            'max_tokens': self.max_tokens,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'n': self.n,
            'stream': self.stream,
            'logprobs': self.logprobs,
            'echo': self.echo,
            'stop': self.stop,
            'presence_penalty': self.presence_penalty,
            'frequency_penalty': self.frequency_penalty,
            'best_of': self.best_of,
            'logit_bias': self.logit_bias
        }.items() if v is not None}
