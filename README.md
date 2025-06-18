# GPU Puzzles on [Modal](https://modal.com/) GPUs

A wrapper around the [GPU puzzles](https://github.com/srush/GPU-Puzzles) by [Sasha Rush](http://rush-nlp.com) ([srush_nlp](https://twitter.com/srush_nlp)), designed to run on [Modal](https://modal.com/) cloud infrastructure.

## Usage

- Install [uv](https://docs.astral.sh/uv/)
- Clone the repo and `cd` into it.
- Run `uv sync` to initialize the virtual environment

Next, create an account with [Modal](https://modal.com/).

To link the modal cli to your account, you can either:

- Follow the onboarding on modal
- Run `uv run modal setup` locally

Once you're setup, run

```
uv run test
```

to execute the puzzle defined in [`src/app.py#L21-23`](src/app.py#L21-23)

## Notes

Everything is hardcoded to on an nvidia T4 [source](src/app.py#L18), the cheapest available instance on modal at this time.
