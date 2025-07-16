import os
import json
from collections import Counter
import adalflow as adal
from typing import Dict
from adalflow.optim.types import ParameterType
from adalflow.components.model_client.openai_client import OpenAIClient
from adalflow.datasets.types import Example
from adalflow.eval.answer_match_acc import AnswerMatchAcc

# ✅ Ensure API key is set
if 'OPENAI_API_KEY' not in os.environ:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# ✅ Model config (GPT-3.5 with deterministic settings)
gpt_3_model = {
    "model_client": OpenAIClient(),
    "model_kwargs": {
        "model": "gpt-3.5-turbo-0125",
        "temperature": 0.0,
        "top_p": 0.95,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "stop": None,
    },
}

# 🧠 Prompt template
few_shot_template = r"""<START_OF_SYSTEM_PROMPT>
{{system_prompt}}
{% if few_shot_demos is not none %}
Here are some examples:
{{few_shot_demos}}
{% endif %}
<END_OF_SYSTEM_PROMPT>
<START_OF_USER>
{{input_str}}
<END_OF_USER>
"""

# ✅ Task Pipeline class
class CountryLetterTaskPipeline(adal.Component):
    def __init__(self, model_client: adal.ModelClient, model_kwargs: Dict):
        super().__init__()

        system_prompt = adal.Parameter(
            data="""You are a reasoning assistant focused on analyzing country names and their letter frequencies.
Determine which country has the highest number of repeated letters. Think step by step.
Your final answer should be in this format: 'Answer: <Country Name>'""",
            role_desc="Guide LLM on how to think and format the answer",
            requires_opt=True,
            param_type=ParameterType.PROMPT,
        )

        few_shot_demos = adal.Parameter(
            data=None,
            role_desc="Few shot examples",
            requires_opt=True,
            param_type=ParameterType.DEMOS,
        )

        self.llm_counter = adal.Generator(
            model_client=model_client,
            model_kwargs=model_kwargs,
            template=few_shot_template,
            prompt_kwargs={
                "system_prompt": system_prompt,
                "few_shot_demos": few_shot_demos,
            },
            use_cache=True,
        )

    def bicall(self, question: str, id: str = None):
        output = self.llm_counter(prompt_kwargs={"input_str": question}, id=id)
        return output

# ✅ Dataset loader with your question
def load_datasets():
    data = [
        {
            "id": "0",
            "input": "What country has the same letter repeated the most in its name?",
            "target": "Saint Vincent and the Grenadines"
        }
    ]

    examples = [Example(id=item["id"], question=item["input"], answer=item["target"]) for item in data]
    return examples, examples, []  # train, val, test

# ✅ AdalComponent class for training
class CountryLetterAdalComponent(adal.AdalComponent):
    def __init__(
        self,
        model_client: adal.ModelClient,
        model_kwargs: Dict,
        backward_engine_model_config: Dict,
        teacher_model_config: Dict,
        text_optimizer_model_config: Dict,
    ):
        task = CountryLetterTaskPipeline(model_client, model_kwargs)
        eval_fn = AnswerMatchAcc(type="exact_match").compute_single_item
        loss_fn = adal.EvalFnToTextLoss(
            eval_fn=eval_fn,
            eval_fn_desc="exact_match: 1 if str(y) == str(y_gt) else 0",
        )
        super().__init__(task=task, eval_fn=eval_fn, loss_fn=loss_fn)

        self.backward_engine_model_config = backward_engine_model_config
        self.teacher_model_config = teacher_model_config
        self.text_optimizer_model_config = text_optimizer_model_config

    def prepare_task(self, sample: Example):
        return self.task.bicall, {"question": sample.question, "id": sample.id}

    def prepare_eval(self, sample: Example, y_pred: adal.GeneratorOutput):
        y_label = -1
        if y_pred is not None and y_pred.data is not None:
            y_label = y_pred.data
        return self.eval_fn, {"y": y_label, "y_gt": sample.answer}

    def prepare_loss(self, sample: Example, pred: adal.Parameter):
        y_gt = adal.Parameter(
            name="y_gt",
            data=sample.answer,
            eval_input=sample.answer,
            requires_opt=False,
        )

        pred.eval_input = pred.full_response.data
        return self.loss_fn, {"kwargs": {"y": pred, "y_gt": y_gt}, "id": sample.id}

# ✅ Training
def train():
    optimizer_model = {
        "model_client": OpenAIClient(),
        "model_kwargs": {"model": "gpt-4o", "temperature": 1.0},
    }

    adal_component = CountryLetterAdalComponent(
        **gpt_3_model,
        teacher_model_config=optimizer_model,
        text_optimizer_model_config=optimizer_model,
        backward_engine_model_config=optimizer_model,
    )

    trainer = adal.Trainer(
        adaltask=adal_component,
        max_steps=4,  # reduce for testing
        raw_shots=1,
        bootstrap_shots=1,
        strategy="random",
    )

    train_dataset, val_dataset, test_dataset = load_datasets()
    trainer.fit(train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset)

    # After training: test the final prompt
    test_sample = train_dataset[0]
    response = adal_component.task.bicall(question=test_sample.question, id="final_test")
    print("\nFinal Prompt Output:")
    print(response.data)

# Run training
if __name__ == "__main__":
    train()
