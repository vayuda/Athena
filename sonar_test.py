# import the sonar encoder + decoder
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
encoder = TextToEmbeddingModelPipeline(encoder="text_sonar_basic_encoder",
                                           tokenizer="text_sonar_basic_encoder")

from sonar.inference_pipelines.text import EmbeddingToTextModelPipeline
decoder = EmbeddingToTextModelPipeline(decoder="text_sonar_basic_decoder",
                                              tokenizer="text_sonar_basic_encoder")

# construct a few sentences
test_sentences = [
    # Garden path sentences
    "The old man the boat with expert skill.",
    "The horse raced past the barn fell.",
    "The complex houses married and single soldiers and their families.",

    # Center-embedded clauses
    "The rat the cat the dog chased killed ate the cheese.",
    "The man who the boy who the students recognized pointed out is a friend.",

    # Complex anaphora
    "John told Bill that he would need his book to study for his exam.",
    "The city councilmen refused the demonstrators a permit because they feared violence.",
    "The trophy doesn't fit in the brown suitcase because it's too big.",

    # Syntactically valid but semantically nonsensical
    "Colorless green ideas sleep furiously.",
    "The square root of my dreams tastes like tomorrow's silence.",

    # Sentences with rare but valid constructions
    "Had had had had had had had had had had had a different meaning.",
    "Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo.",
    "That that is is that that is not is not is that it it is.",

    # Multi-step temporal reasoning
    "If Tom visited Paris before Mary went to Rome, but after Susan left London, and Susan was in London during Mary's birthday, when did Tom visit Paris relative to Mary's birthday?",
    "Before the meeting started after lunch, which was delayed because of traffic that began before dawn, when did the participants actually arrive?",

    # Counterfactuals with nested conditions
    "If the lights hadn't turned red when they did, but instead had stayed green until after the truck passed, the accident that didn't actually happen would have occurred, unless the driver who wasn't speeding had been.",
    "In a world where silence was loud and noise was quiet, if whispering became shouting, would a library still require hushed voices?",

    # Mathematical reasoning in natural language
    "In a room where everyone shook hands with everyone else exactly once, and there were 45 total handshakes, how many people must have been in the room?",
    "If twice the difference between half of a number and a third of that same number equals the number itself, what is the number?",

    # Analogical reasoning
    "A fish is to water as a bird is to air, but a penguin is to water as a ostrich is to land, so a flying fish is to air as a seal is to what?",
    "If knowledge is to wisdom as data is to information, then raw numbers are to statistics as experience is to what?",

    # Causal chains
    "The factory shutdown caused supply shortages, which led to price increases, unless government intervention stabilized markets, but only if implemented before distributors adjusted their inventory strategies.",
    "The drought affected crop yields, which impacted farmer incomes, leading to reduced local spending, causing small business closures, ultimately resulting in population decline, unless alternative industries emerged.",

    # Mixed complexity
    "The fact that the solution that the team that the manager hired proposed solved the problem that the client who the CEO called complained about impressed everyone.",
    "If what he told me about what you said regarding what they believed about what actually happened is true, then someone must be mistaken about what really occurred.",
]

from datasets import load_dataset
extraction_size = 30

dataset = load_dataset("wiki_news")
scientific = load_dataset("scientific_papers", "arxiv")
wikihow = load_dataset("wikihow", "all")

wiki_samples = dataset['train'][:extraction_size]
wikihow_samples = wikihow['train'][:extraction_size]
abstracts = scientific['train']['abstract'][:30]


# filter/sample specific sentences
# long_sentences = [x['text'] for x in wikihow['train'] if len(x['text'].split()) > 30][:10]
# medium_sentences = [x['text'] for x in wikihow['train'] if 15 <= len(x['text'].split()) <= 30][:10]
# short_sentences = [x['text'] for x in wikihow['train'] if len(x['text'].split()) < 15][:10]






embeddings = encoder.predict(test_sentences, source_lang="eng_Latn")
reconstructions = decoder.predict(embeddings, target_lang="eng_Latn", max_seq_len=512)
for i in range(len(test_sentences)):
    print("reconstructed: ", reconstructions[i])
    print("ground truth: ", test_sentences[i])
