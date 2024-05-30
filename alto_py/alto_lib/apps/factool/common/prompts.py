# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 17:00:39 2023

@author: Shahir
"""

QUESTION_ANSWERING_PROMPT = "\n".join(
    [
        "You are given a question which can be answered using factual knowledge of the world.",
        "Your task is to answer the given question. The question will include a specified topic which will help give context for your response.",
    ]
)

CLAIM_EXTRACTION_PROMPT = "\n".join(
    [
        "You are given a piece of text that includes knowledge claims. A claim is a statement that asserts something as true or false, which can be verified by humans.",
        "[Task]",
        "Your task is to accurately identify and extract every claim stated in the provided text. Then, resolve any coreference (pronouns or other referring expressions) in the claim for clarity. Each claim should be concise (less than 15 words) and self-contained.",
        "Your response MUST be a single string per line. Each line must be only the extracted claim (with all coreferences resolved). You MUST only respond in the format as described below. DO NOT RESPOND WITH ANYTHING ELSE. ADDING ANY OTHER EXTRA NOTES THAT VIOLATE THE RESPONSE FORMAT IS BANNED. ",
        "[Response Format]",
        '"Ensure that the claim is fewer than 15 words and conveys a complete idea.',
        '"Resolve any coreference (pronouns or other referring expressions) in the claim for clarity."',
        "Here are two examples:",
        "[text]:",
        "Tomas Berdych defeated Gael Monfis 6-1, 6-4 on Saturday. The sixth-seed reaches Monte Carlo Masters final for the first time. Berdych will face either Rafael Nadal or Novak Djokovic in the final.",
        "[response]:",
        "Tomas Berdych defeated Gael Monfis 6-1, 6-4",
        "Tomas Berdych defeated Gael Monfis 6-1, 6-4 on Saturday",
        "Tomas Berdych reaches Monte Carlo Masters final",
        "Tomas Berdych is the sixth-seed",
        "Tomas Berdych reaches Monte Carlo Masters final for the first time",
        "Berdych will face either Rafael Nadal or Novak Djokovic",
        "Berdych will face either Rafael Nadal or Novak Djokovic in the final",
        "[text]:",
        "Tinder only displays the last 34 photos - but users can easily see more. Firm also said it had improved its mutual friends feature.",
        "[response]:",
        "Tinder only displays the last photos",
        "Tinder only displays the last 34 photos",
        "Tinder users can easily see more photos",
        "Tinder said it had improved its feature",
        "Tinder said it had improved its mutual, friends feature",
    ]
)

QUERY_GENERATION_PROMPT = "\n".join(
    [
        "You are a query generator designed to help users verify a given claim using search engines.",
        "[Task]",
        "Your primary task is to generate effective and skeptical search engine queries. These queries should assist users in critically evaluating the factuality of a provided claim using search engines. You should only respond in format as described below.",
        "Your response MUST be a single string per line. Each line must only be a search engine query. PLEASE STRICTLY FOLLOW THE FORMAT. DO NOT RETURN ANYTHING ELSE.",
        "[Response Format]",
        "The query should be used to verify the provided claim.",
        "The query should include useful terms from the claim.",
        "Here are 3 examples:",
        "[claim]:",
        "The CEO of twitter is Bill Gates.",
        "[response]:",
        "Who is the CEO of twitter?",
        "CEO Twitter",
        "[claim]:",
        "Michael Phelps is the most decorated Olympian of all time.",
        "[response]:",
        "Who is the most decorated Olympian of all time?",
        "Michael Phelps",
        "[claim]:",
        "ChatGPT is created by Google.",
        "[response]:",
        "Who created ChatGPT?",
        "ChatGPT",
    ]
)


CLAIM_APPRAISAL_PROMPT = "\n".join(
    [
        "You are given a piece of text. Your task is to identify whether there are any factual errors within the text. When you are judging the factuality of the given text, you could reference the provided evidences if needed. The provided evidences may be helpful. Some evidences may contradict to each other. You must be careful when using the evidences to judge the factuality of the given text. When The response should be a dictionary with four keys - 'reasoning', 'factuality', 'error', and 'correction', which correspond to the reasoning, whether the given text is factual or not (Boolean - True or False), the factual error present in the text, and the corrected text. You should only respond in format as described below. DO NOT RETURN ANYTHING ELSE. START YOUR RESPONSE WITH '{{'.",
        "[response format]:",
        "{{ 'reasoning': 'Why is the given text factual or non-factual? Be careful when you said something is non-factual. When you said something is non-factual, you must provide mulitple evidences to support your decision.', 'error': 'None if the text is factual; otherwise, describe the error.', 'correction': 'The corrected text if there is an error.', 'factuality': True if the given text is factual, False otherwise. }}",
    ]
)

CLAIM_APPRAISAL_PATTERNS = {
    "reasoning": r'["\']reasoning["\']\s*:\s*[\'"](?P<reasoning>.+?)[\'"],',
    "error": r'["\']error["\']\s*:\s*[\'"]?(?P<error>.+?)[\'"]?,',
    "correction": r'["\']correction["\']\s*:\s*[\'"]?(?P<correction>.+?)[\'"]?,',
    "factuality": r'["\']factuality["\']\s*:\s*(?P<factuality>True|False|\'True\'|\'False\')',
}
