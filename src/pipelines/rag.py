import markdown
from metaflow import Config, Parameter, card, step

from agents.rag.agent import Agent
from common.pipeline import Pipeline


def read_template(html):
    """Parse the supplied HTML template.

    This function is used to read an HTML template from a Metaflow Config
    parameter and return it as a dictionary that can be used by the card
    decorator.
    """
    return {"html": html}


class Rag(Pipeline):
    """A Metaflow pipeline that answers questions using a RAG agent."""

    model = Parameter(
        name="model",
        help="The underlying model that will be used by the agent.",
        default="gemini/gemini-2.5-flash",
    )

    template = Config("template", default="config/rag.html", parser=read_template)

    @card
    @step
    def start(self):
        """Start the pipeline by defining the questions we want to ask."""
        self.questions = [
            "Summarize how Metaflow branches work.",
            "How do I run the Training pipeline locally?",
            "Where can I find the code for the Training pipeline?",
        ]

        # For each question, we will use the agent to get an answer.
        self.next(self.answer_question, foreach="questions")

    @step
    def answer_question(self):
        """Run the agent to answer the supplied question."""
        # Let's create an instance of the agent that we want to use to answer the
        # question and initialize it with the supplied model.
        agent = Agent(model=self.model, logger=self.logger)

        self.question = self.input
        self.response = agent.run(question=self.question)

        self.status = self.response["status"]
        self.answer = self.response.get("answer", "")

        self.next(
            {
                "success": self.success,
                "failed": self.failed,
            },
            condition="status",
        )

    @card(type="html")
    @step
    def success(self):
        """Join the parallel branches and create the final response."""
        self.html = self.template["html"]

        # We want to convert the answer from Markdown to HTML before injecting it
        # into the HTML template.
        try:
            answer = markdown.markdown(
                self.answer,
                extensions=["fenced_code", "tables", "codehilite", "toc", "sane_lists"],
            )
        except Exception:
            # If the conversion fails for any reason, we will just use the original
            # answer.
            answer = self.answer

        self.html = self.html.replace("[[QUESTION]]", self.question)
        self.html = self.html.replace("[[ANSWER]]", answer)

        self.next(self.join)

    @step
    def failed(self):
        """Handle failures in the agent."""
        self.logger(f'Failed to answer question "{self.question}".')
        self.next(self.join)

    @step
    def join(self, inputs):
        """Join the parallel branches and create the final response."""
        self.response = ""
        for question, answer in [(i.question, i.answer) for i in inputs]:
            self.response += f"Question: {question}\nAnswer: {answer}\n\n"

        self.next(self.end)

    @step
    def end(self):
        """End the pipeline by printing the final response."""
        self.logger.info("%s", self.response)


if __name__ == "__main__":
    Rag()
