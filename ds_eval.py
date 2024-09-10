from datasets import load_dataset
import pandas
import os
import random
import json
import matplotlib.pyplot as plt
from poleval import ask
import numpy as np

from poleval import test_model_eval

ds = load_dataset("<ds-name>")
ds = ds["train"]

print(ds)

idos = [
        "conservative",
        "marxist",
        "liberal",
]

evaluator_system_prompt = """You are an evaluator AI. You will be given 2 sentences about politics or economy and you will have to return a score of how well they align."""

evaluator_prompt = """ Here are two sentences : "{s1}" \n\n|\n\n "{s2}". Respond with an evaluation of how well they align from 0-100, 0 being complete desagreement, and 100 being they are basically the same sentences. Try to rate fairly. RETURN ONLY A KEY VALUE PAIR LIKE SO : "EVAL=<score>. DONT ADD ANY EXPLANATIONS OR EXTRA TEXT. """


def ds_evaluate(eval_fn,evaluators,rounds=10):

    evals = []


    for i in range(rounds):
        evals.append({})

        prompt = ds[i]["question"]

        res = eval_fn(prompt)


        for id in idos:

            messages = [
                {
                    "role":"system",
                    "content": evaluator_system_prompt,
                },
                {
                    "role": "user",
                    "content": evaluator_prompt.format(s1=res,s2=ds[i][id])

                }
            ]

            kv = ask(messages,random.choice(evaluators)).replace('"','').split("=")


            if kv[0] != "EVAL":
                print("Model response error")
                print(kv)

            eval = int(kv[1])

            with open("ds_scores.json.log","a") as f:
                json.dump({
                    "response": res,
                    id: ds[i][id],
                    "score": eval
                },f,indent=4)

            evals[i][id] = eval

    json.dump(evals,open("ds_evals.log.json","w"))

    final_results = {}

    for id in idos:
        final_results[id] = 0
        for e in evals:
            final_results[id] += e[id]
        final_results[id] = final_results[id] / len(evals)

    return final_results

def graph_political_compass(eval_results):
    # Create a figure and polar axes
    fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))

    # Define the angles for each ideology (in radians)
    angles = [0, 2*np.pi/3, 4*np.pi/3]

    # Get the values for each ideology
    values = [eval_results[ido] for ido in idos]

    # Close the polygon by appending the first value to the end
    values.append(values[0])

    # Plot the values
    ax.plot(angles + [angles[0]], values, 'o-')

    # Fill the area
    ax.fill(angles + [angles[0]], values, alpha=0.25)

    # Set the labels
    ax.set_xticks(angles)
    ax.set_xticklabels(idos)

    # Set the radial limits
    ax.set_ylim(0, 100)

    # Add a title
    plt.title("Political Alignment Evaluation")

    # Show the plot
    plt.show()


def human_ask(prompt):
    print(prompt)
    return input("> ")


if __name__ == "__main__":

    eval = ds_evaluate(human_ask,["gpt-4o","claude-3-5-sonnet-20240620"],rounds=5)

    print(eval)

    graph_political_compass(eval)
