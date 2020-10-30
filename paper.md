# An Ai that can Do your Homework

This is the dream of all 10 year olds. Sitting in their bed, not wanting to go to school (this is me. I'm all 10 year olds). Well guess what, it's easier than ever to make, and here's how I didit, it's as simple as 1, 2, 3!

- Collect Data
- Train the AI
- Use it

Just kidding.

In all reality, though, we're not far off from just plugging data into a random app and having our cusomized ai. I'm sure someone has already done it. We're gonna need to do a little bit of coding though. (edit, [this](https://deepcognition.ai/) seems like the closest thing to that at the moment)

We will be using these as our main steps to complete.

I used the GPT-2 model from [OpenAI](https://openai.com/blog/better-language-models/) to make my model. I also used a library from Max Woolf and his Google Colaboratory notebook. [Found here](https://github.com/minimaxir/gpt-2-simple).

> Developed by OpenAI, **GPT**-**2** is a pre-trained language model which we can use for various NLP tasks, such as: Text generation. Language translation. Building question-answering systems, and so on - [Shubham Singh](https://www.analyticsvidhya.com/blog/author/shubham-singh/)

This notebook has all the things we need to train and run the model, except for the data.

I used a there's a dataset called SQUAD, which is all about reading comprehension and question answering, exactly what we need. So I found a CSV form of the SQUAD dataset. [Here.](https://www.kaggle.com/ananthu017/squad-csv-format) If you want to explore the original, [here's a link](https://rajpurkar.github.io/SQuAD-explorer/) for that.

Let's go through what we have so far,

- Data
- A way to train the model
- and a way to use it.

That's everything we need right?

Wrong!

## Cleaning the Data

- Take data from csv
- Put it together
- Save as file

We need to clean up the csv file and transform it into a `.txt` file. Take the `context`, `question`, and `answer` parts of it and iteratively write them to the file. (I already did so and the file can be found [here](https://gist.github.com/spronkoid/e721c592b98384923db9a6df4d6cf5e5)). Here's how I did it.

```python
import pandas
colnames = ['index', 'text', 'questions', 'answers']
data = pandas.read_csv('QA Dataset.csv', names=colnames)
```

The first line imports the library we're going to use. (A library is a set of commands that we can use instead of writing them ourselves.)

The second line is how we tell the program what columns there are in the CSV file, and the last line reads the csv file so we can work with it.

Now we need to compile all the contexts, questions, and answers together, but with a `<|startoftext|>` and `<|endoftext|`> to denote them. We also need to clean up the answers because they're formatted in a way where GPT-2 (the model we're using) has to work a little extra hard to understand.

```python
text = data.text.tolist()
questions = data.questions.tolist()
answers = data.answers.tolist()

c, q, a, e = "<|startoftext|>\n[CONTEXT]: ", "\n[QUESTION]:", "\n[ANSWER]:", "\n<|endoftext|>\n"
index = 1

while(index < len(text)):
    dataset += c+text[index]

    qindex = 1

    while qindex < len(eval(questions[index])):
        question = eval(questions[index])[qindex]
        questionS = ''.join(map(str, question))
        answer = eval(answers[index])[qindex]
        answerS = ''.join(map(str, answer))

        answerS = answerS[:-21:] # removes the last 21 characters of the answer
        answerS = answerS[9::]  # removes the first 9 characters of the answer

        dataset += q+questionS
        dataset += a+answerS

dataset += e
index += 1

print(dataset)
```

That was the quickest way I could think of doing it, I'm not a programming mastermind. It works exactly how we need it to. Now we can write it all into a `.txt` file with these few lines:

```python
f = open("QAdataset.txt", "w+")
f.write(dataset)
f.close()
```

## Setting up the Model

Cool! Now we have everything we need. Let's open up [Max Woolf's notebook](https://colab.research.google.com/drive/1VLG8e7YSEwypxU-noRNhsv5dW4NfTGce) and get to work.

All we have to do is change this:

```python
gpt2.download_gpt2(model_name="124M")
```

to this:

```python
gpt2.download_gpt2(model_name="345M")
```

so that we use a more powerful model, and then we change our file name (he suggests we upload it to drive, we don't have to though and I just didn't run the line of code after this:)

```python
file_name = "shakespeare.txt"
```

instead of `"shakespeare.txt"`, we put `"QAdataset.txt"`.

## Training the Model

The last thing we change is the number of steps to train it for, and maybe the model name if you want. Here's what mine looked like:

```python
gpt2.finetune(
    sess,
    dataset=file_name,
    model_name='345M',
    steps=2000,
    restore_from='fresh',
    run_name='QA',
    print_every=1,
    sample_every=2000,
    save_every=100
)
```

and we're complete! My final model ended with a loss of 0.06 after 4000 steps.

## Using it

When generating answers, I wrote this bit of code:

```python
ctx = input("Context: ")
qst = input("Question: ")
ans = "[ANSWER]:"
pre = '<|startoftext|>\n[CONTEXT]: ' + ctx + "\n[QUESTION]:" + qst + "\n" + ans
```

to make it a bit easier to format the prompt. The context is data about your subject, I've been testing it on the first paragraph of wikipedia pages. The question is what you want to ask about that paragraph. You can `print(pre)` to copy and paste it, or just pass `pre` as the prompt instead of a string. Here's what my code looked like for generating,

```python
gpt2.generate(
    sess,
    length=5,
    temperature=0.7,
    prefix=pre,
    nsamples=10,
    batch_size=10,
    run_name="QA",
)
```

We make it generate 10 times, because it's not always accurate, so we can see which one is the correct answer by seeing which thing it answered the most. If it doesn't generate your full answer then crank up the length to around 10 or 20 (it's how much the model predicts)

## El Fin

If you want the notebook I used it'll be here: [link](https://colab.research.google.com/drive/1hGxYPTx0E515cPcY-_vWfsdHMJQ1hPP8)

If you just want my trained model to add to your drive, it can be found here: [link](https://drive.google.com/open?id=1vECDEYNRaH7Y785_YFDWiNpmnQde0_7O). Though you'll have to change everything from `QA` to `QA3-test2`. This was my third time trying a new way to do it, and second version of the third time.

This took me a long while to get exactly how I wanted. The first time and second times I didn't take the first and last few characters off of the answers in the dataset, and it only ever got down to 0.19 loss no matter how hard I tried.

After I did chop them off though, it got down to 0.06 in around an hour or so (4000 iterations).

here's what the last step looked like: `loss=0.05 avg=0.06`

last step of the previous try had: `loss=0.20 avg=0.19`

ALWAYS clean your data, it makes a big deal.

How can this be improved?

- writing a script to automatically take the answer it predicts the most and display that

- try huggingface's pytorch implementation and add or switch some last layers for ones better suited for question answering
- see if any other delimiters work better

Lessons I learned from this:

- Cleaning your data makes a <u>**_huge_**</u> difference
- Language models aren't exceptional at numbers
- Taking a little extra time to finish something is the _best thing you can do_

and finally:

> “Any work is always improvable, you cannot really finish the work, you can only abandon it out of tiredness or incompetence.” ― Amit Kalantri

I'm writing this paper because I've abandoned the project out of incompetence for now. I have no means to improve it _myself_.

That's all for now.

Thanks for reading,

[@samuelreams](twitter.com/spronkoid)
