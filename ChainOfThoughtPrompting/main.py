import os
from datetime import datetime
from time import sleep

from google import genai
from dotenv import load_dotenv
import json
import ollama

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, ".env")

load_dotenv(file_path)
GEMINI_API_KEY = os.getenv("GENAI_API_KEY")


gemini_client = None
if GEMINI_API_KEY:
  try:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
  except Exception as e:
    print(f"Gemini couldn't be initialized. {e}")

benchmark_prompt = 'Important: Provide a JSON object AT THE VERY END of your answer containing only the final answer as a number. Use this JSON schema: { "answer": <The correct answer as a number> }'

query_examples = {
  "tony": "Tony has 5 Apples. His friend Anthony has another 3 Apples. His mother asked him to collect a sum of 10 Apples before returning home. How many Apples are left for Tony to collect?",
  "gamer": "A gamer has 2 graphics cards in his computer. He buys two more graphics cards, but gifts one to a friend. How many graphics cards does he have?"
}

benchmark_questions = [
    {
        "question": "A farmer has 12 cows in his barn. He buys 8 more but sells 5 to a neighboring farmer. How many cows does he have now?",
        "answer": 12 + 8 - 5
    },
    {
        "question": "Sarah has 3 apples. She buys a dozen more but eats 2. How many apples does she have left?",
        "answer": 3 + 12 - 2
    },
    {
        "question": "A library has 500 books. They receive a donation of 250 books but remove 100 damaged books. How many books are in the library now?",
        "answer": 500 + 250 - 100
    },
    {
        "question": "A factory produces 40 chairs per day. How many chairs does it produce in a week if it operates 6 days a week?",
        "answer": 40 * 6
    },
    {
        "question": "John runs 5 kilometers every morning. If he runs every day for 2 weeks, how many kilometers does he run in total?",
        "answer": 5 * 7 * 2
    },
    {
        "question": "A train can carry 80 passengers per carriage. If the train has 10 carriages, how many passengers can it carry in total?",
        "answer": 80 * 10
    },
    {
        "question": "A bakery bakes 120 loaves of bread every morning. By noon, they have sold 85 loaves. How many loaves are left?",
        "answer": 120 - 85
    },
    {
        "question": "A football team scores 3 goals per match. How many goals will they have scored after playing 8 matches?",
        "answer": 3 * 8
    },
    {
        "question": "A hotel has 250 rooms, but 40 are under renovation. If 175 rooms are occupied, how many rooms are available for new guests?",
        "answer": 250 - 40 - 175
    },
    {
        "question": "A school orders 600 pencils. They distribute 15 pencils per classroom, and there are 30 classrooms. How many pencils are left over?",
        "answer": 600 - 15 * 30
    },
    {
        "question": "A software company hires 15 developers in January and 10 more in February. If 5 developers resign in March, how many developers remain?",
        "answer": 15 + 10 - 5
    },
    {
        "question": "A water tank holds 1,000 liters of water. A household consumes 150 liters per day. How much water is left after 4 days?",
        "answer": 1000 - 150 * 4
    },
    {
        "question": "A cyclist completes a 20-kilometer race 4 times a week. How many kilometers does he cycle in a month with 5 weeks?",
        "answer": 20 * 4 * 5
    },
    {
        "question": "A bookstore sells 25 books per day. If they have 500 books in stock, how many days will it take to sell out completely?",
        "answer": 500 / 25
    },
    {
        "question": "A plane has 180 passenger seats, but 25 are unoccupied. How many passengers are on board?",
        "answer": 180 - 25
    },
    {
        "question": "A concert venue has 1,500 seats. If 80% of the tickets are sold, how many seats remain empty?",
        "answer": 1500 * 0.2
    },
    {
        "question": "A gardener plants 45 flowers in a row. If he makes 8 rows, but removes 9 flowers how many flowers are planted in total?",
        "answer": 45 * 8 - 9
    },
    {
        "question": "A supermarket sells 35 bottles of juice per day. One bottle of juice costs 3 dollars. How much money will they make in a 12-day period?",
        "answer": 35 * 3 * 12
    },
    {
        "question": "A swimming pool can hold 10,000 liters of water. If 2,500 liters evaporate in a week, how much water remains after 3 weeks?",
        "answer": 10000 - 2500 * 3
    },
    {
        "question": "A sports shop sells 3 basketballs for $60. How much will 7 basketballs cost at the same rate?",
        "answer": 60 / 3 * 7
    },
    {
        "question": "Tony is collecting apples per his mother's request. He already has 5 apples. His friend Anthony has 3 apples. Tony needs to collect 10 apples in total for his mother. However, the apples collected from Anthony do not belong to Tony. How many more apples does Tony need to collect before returning home?",
        "answer": 10 - 5
    }
]


cot_few_shot_examples = [
    {
        "question": "Tony has 5 Apples. His friend Anthony has another 3 Apples. His mother asked him to collect a sum of 10 Apples before returning home. How many Apples are left for Tony to collect?",
        "reasoning": {
            "text": "Tony currently has 5 apples while his friend has 3 apples. He needs a total of 10 apples. 10 - 5 - 3 = 2. Anthony needs to collect two more apples before returning home.",
            "bulletPoints": """
            - Tony has 5 apples.
            - His friend Anthony has 3 apples.
            - The total number of apples Tony needs to collect is 10.
            - 10 - 5 - 3 = 2.
            - Tony needs to collect 2 more apples before returning home.
            """
        },
        "answer": 10 - 5 - 3
    },
    {
        "question": "A gamer has 2 graphics cards in his computer. He buys two more graphics cards, but gifts one to a friend. How many graphics cards does he have?",
        "reasoning": {
            "text": "The gamer starts with 2 graphics cards. He buys 2 more, but gives one away. 2 + 2 - 1 = 3. The resulting number of graphics card the gamer has is 3.",
            "bulletPoints": """
            - The gamer has 2 graphics cards.
            - He buys 2 more graphics cards.
            - He gifts one to a friend.
            - 2 + 2 - 1 = 3.
            - The gamer has 3 graphics cards.
            """
        },
        "answer": 2 + 2 - 1
    },
    {
        "question": "A farmer has 5 cows. He buys 3 more cows. However, 2 of the cows run away. How many cows does the farmer have now?",
        "reasoning": {
            "text": "The farmer has 5 cows he owns. He purchases 3 more cows, which increases his total amount of cows. 5 + 3 = 8. The farmer has 8 cows. However, 2 of the cows run away. 8 - 2 = 6. The farmer has 6 cows left.",
            "bulletPoints": """
            - The farmer has 5 cows.
            - He buys 3 more cows.
            - 5 + 3 = 8.
            - The total number of cows is 8.
            - 2 of the cows run away.
            - 8 - 2 = 6.
            - The farmer has 6 cows left.
            """
        },
        "answer": 5 + 3 - 2
    },
    {
        "question": "A teacher has 33 students in her class. 3 Students leave her class early. She decides to divide the students into 6 groups. How many students are in each group?",
        "reasoning": {
            "text": "The teacher has 33 students in her class in total, although 3 decide to leave. 33 - 3 = 30. The remaining number of students is 30. She divides the remaining students into 6 groups. 30 / 6 = 5, the remaining number of students equals 5. There are 5 students per group.",
            "bulletPoints": """
            - The teacher has 33 students.
            - 3 students leave her class.
            - 33 - 3 = 30.
            - The remaining number of students is 30.
            - She divides them into 6 groups.
            - 30 / 6 = 5.
            - Each group has 5 students.
            """
        },
        "answer": (33 - 3) / 6
    },
    {
        "question": "A driver went on a drive which consumed 10 liters worth of petrol. The driver decides to take a fuelstop to fill his car with 25 liters of petrol. If the car's tank can hold 21 liters of petrol, how many liters of petrol will overflow?",
        "reasoning": {
            "text": "The driver went on a drive consuming 10 liters. However the question is not about the fuel consumption. The driver decides to fill his car with 25 out of 21 liters of petrol the car can hold. 21 - 25 = -4. The question is instead about the amount of petrol that will overflow, overflow being defined here as resulting negative values. The amount of petrol that will overflow from his tank is equal to 4 liters.",
            "bulletPoints": """
            - The driver consumed 10 liters of petrol.
            - The question is not about fuel consumption.
            - The driver fills his car with 25 liters of petrol.
            - The car's tank can hold 21 liters of petrol.
            - 21 - 25 = -4.
            - The question is instead about the amount of petrol that will overflow, overflow being defined here as resulting negative values.
            - The amount of petrol that will overflow is 4 liters.
            """
        },
        "answer": (21 - 25)
    },
    {
        "question": "There are four cats sitting on a rooftop. Each cat has 4 legs. How many legs are there in total?",
        "reasoning": {
            "text": "Each cat has 4 legs. There are 4 cats in total. 4 * 4 = 16. The total number of legs is 16.",
            "bulletPoints": """
            - Each cat has 4 legs.
            - There are 4 cats.
            - 4 * 4 = 16.
            - The total number of legs is 16.
            """
        },
        "answer": 4 * 4
    }
]

def is_gemini_available():
    return gemini_client is not None

def is_llama_available():
    try:
        ollama_models = [x.model for x in ollama.list().models]
        for ollama_model in ollama_models:
            if ollama_model.startswith("llama3.2"):
                return True
    except:
        return False
    return False


def ask_gemini(prompt, do_cooldown=False):
    response = gemini_client.models.generate_content(model="gemini-1.5-pro", contents=prompt).text
    if do_cooldown:
        print("🥵 Cooling down Gemini")
        sleep(15)
        print("🥶 Gemini cooled down. Continuing...")
    return response

def ask_llama(prompt):
    return ollama.generate(
        model="llama3.2",
        prompt=prompt
    )["response"]

def ask_chatbot(prompt, ai_model, do_cooldown=False):
    if ai_model == "Gemini 1.5 Pro":
        return ask_gemini(prompt, do_cooldown)
    elif ai_model == "llama3.2":
        return ask_llama(prompt)

def create_cot_zero_shot_prompt(user_prompt, benchmark=False):
    return f"""
    Imagine your are an advanced AI Assistant that follows Chain of Thought (CoT) reasoning principle.
    You are asked to answer the following question:
    
    Question: {user_prompt}
    
    Let's think step by step:
    
    {benchmark_prompt if benchmark else ""}
    """

def create_cot_few_shot_prompt(user_prompt, examples, benchmark=False):
    prompt = """
    You are an advanced AI that follows Chain of Thought (CoT) reasoning.
    
    """

    for i, example in enumerate(examples):
        prompt += f"""
        Example {i + 1}:
        Question: {example["question"]}
        Let's think step by step:
        {example["reasoning"]["text"]}
        Answer: {example["answer"]}
        
        """

    prompt += f"""
    
    Now, answer this question: {user_prompt}
    
    Let's think step by step.
    
    {benchmark_prompt if benchmark else ""}
    """
    return prompt

def create_non_cot_prompt(user_prompt, benchmark=False):
    return f"""
    {user_prompt}
    
    {benchmark_prompt if benchmark else ""}
    """

def parse_answer_from_benchmark_response(response):
    num = ''
    is_digit_detected = False

    for char in reversed(response):
        if char.isdigit():
            is_digit_detected = True
            num = char + num
        elif num:
            if is_digit_detected:
                break
    try:
        return float(num)
    except:
        return None

def analyse_benchmark_results(ai_model, benchmark_results):
    amount_of_correct_non_cot_answers = sum(result["isNonCotAnswerCorrect"] for result in benchmark_results)
    amount_of_correct_cot_answers = sum(result["isCotAnswerCorrect"] for result in benchmark_results)
    amount_of_benchmarks = len(benchmark_results)

    return {
        "aiModel": ai_model,
        "amountOfCorrectNonCotAnswers": amount_of_correct_non_cot_answers,
        "amountOfCorrectCotAnswers": amount_of_correct_cot_answers,
        "amountOfBenchmarks": amount_of_benchmarks,
        "successRateNonCotAnswers": amount_of_correct_non_cot_answers / amount_of_benchmarks,
        "successRateCotAnswers": amount_of_correct_cot_answers / amount_of_benchmarks,
        "results": benchmark_results
    }

def save_benchmark_analysis(benchmark_analysis):
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    ai_model = benchmark_analysis["aiModel"].lower().replace(" ", "-")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, "benchmarks", f"{ai_model}-{timestamp}.json")

    with open(file_path, "w") as file:
        file.write(json.dumps(benchmark_analysis, indent=4))


def run_benchmark(ai_model, max_benchmark_questions=None):
    results = []

    if max_benchmark_questions is None or max_benchmark_questions >= len(benchmark_questions):
        benchmark = benchmark_questions
    else:
        benchmark = benchmark_questions[:max_benchmark_questions]

    for i, benchmark_question in enumerate(benchmark):
        print(f"Running benchmark {i+1}/{len(benchmark)}")

        non_cot_response = ask_chatbot(
            create_non_cot_prompt(benchmark_question["question"], True),
            ai_model,
            True
        )
        is_non_cot_response_correct = benchmark_question["answer"] == parse_answer_from_benchmark_response(non_cot_response)

        cot_response = ask_chatbot(
            create_cot_few_shot_prompt(benchmark_question["question"], cot_few_shot_examples[:4], True),
            ai_model,
            True
        )
        is_cot_response_correct = benchmark_question["answer"] == parse_answer_from_benchmark_response(cot_response)

        results.append({
            "question": benchmark_question["question"],
            "correctAnswer": benchmark_question["answer"],
            "nonCotResponse": non_cot_response,
            "isNonCotAnswerCorrect": is_non_cot_response_correct,
            "cotResponse": cot_response,
            "isCotAnswerCorrect": is_cot_response_correct
        })

        if ai_model == "Gemini 1.5 Pro":
            sleep(65) # Cooldown, so minute based quota is not violated on Gemini API

    return analyse_benchmark_results(ai_model, results)


def run_and_print_benchmark():
    print(json.dumps(run_benchmark("llama3.2"), indent=4))


def ask_mathbot(config):
    prompt = config["userPrompt"]

    if config["useCot"]:
        if config["cotMode"] == "Zero-Shot":
            prompt = create_cot_zero_shot_prompt(config["userPrompt"])
        elif config["cotMode"] == "Few-Shot":
            prompt = create_cot_few_shot_prompt(config["userPrompt"], cot_few_shot_examples[:4])

    response = ask_chatbot(prompt, config["aiModel"])

    explicit_non_cot_response = None

    if config["generateAdditionalNonCotResponse"]:
        explicit_non_cot_response =  ask_chatbot(config["userPrompt"], config["aiModel"])

    return {
        "response": response,
        "explicitNonCotResponse": explicit_non_cot_response
    }


if __name__ == "__main__":
    # run_and_print_benchmark()
    print(
        #parse_answer_from_benchmark_response("hellos dfhiusdhfiusdf 12 udshfiudf")
        is_llama_available()
    )
