import os
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
    # code
    "x=a+b",
    "return fibonacci(n-1) + fibonacci(n-2) if n>1 else n",
    "numbers = [i for i in range(10)]",
    #math
    r"\begin{pmatrix} a & b \\ c & d \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix}",
    r"\frac{-b \pm \sqrt{b^2 - 4ac}}{2a} = \sum_{n=0}^{\infty} \frac{x^n}{n!} \cdot \prod_{i=1}^n \frac{1}{i}",
    #science
    'this approximation can only be valid if applied to fluctuations that involve a sufficiently large number of molecules .',
 "electroweak corrections to longitudinal gauge and higgs boson scattering amplitudes are calculated .",
 "in other words , the helical world chain describes the particle rotation with superluminal velocity .",
"we show that this operation is compatible with the mutation operation on the tilting objects",
    "our results confirm that the density of states has a step - like shape but no square - root singularity at the spectrum onset .",
    #law
    "As we recognized in INS v. Cardoza-Fonseca, 480 U.S. 421, 428-429, 107 S.Ct. 1207, 1211-1212, 94 L.Ed.2d 434 (1987)",
     "Section 243(h) of the INA, as amended, provides that, subject to four enumerated exceptions:   30",
    "In June 1988, Attorney General Meese reversed the BIA and ordered respondent deported to the United Kingdom.  ",
     "The Court of Appeals for the Second Circuit reviewed both the order of Attorney General Meese which denied respondent's designation of Ireland as the country of deportation and Attorney General Thornburgh's order denying respondent's motion to reopen his deportation proceeding.",
     "The Attorney General here held that respondent\'s decision to withdraw certain claims in the initial proceedings was a 'deliberate tactical decision,' and that under applicable regulations those claims could have been submitted at that time even though inconsistent with other claims made by the respondent.",
    #tiny stories
    "Emma wanted to show off her new comb, so she decided to act brave and walk to the park with it.",
 "She learned that even if you are impatient, good things can happen if you wait and take care of them.",
 "The moral of the story is that when you learn something new, you should use your skills to help others and make the world a better place.",
 "One day, a funny cat named Kitty wanted to organize a play day for all her friends.",
 "He used his paw to remove the balloon from the ceiling.",
]

def auto_encode(text):
    embeddings = encoder.predict(text, source_lang="eng_Latn")
    return decoder.predict(embeddings, target_lang="eng_Latn", max_seq_len=512)
def eval_from_directory(directory):
    """Reads sentences from files in a directory and returns them as a list."""
    all_sentences = {}
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                sentences = [line.strip() for line in file]
                reconstructions = auto_encode(sentences)
                score = sum([i == j for i, j in zip(sentences, reconstructions)])/len(sentences)
                all_sentences[filename] = (sentences, reconstructions, score)
                
    return all_sentences
    
results = eval_from_directory("sentences")
for i, (sentences, reconstructions, score) in results.items():
    print(f"Results for {i}:")
    # print(f"Original: {sentences}")
    with open(f'sentences/reconstructions_{i}', 'w+') as f:
        for sentence, reconstruction in zip(sentences, reconstructions):
            f.write(f"Ground truth: {sentence}\n")
            f.write(f"Reconstructed: {reconstruction}\n\n")
    # print(f"Reconstructed: {reconstructions}")
    print(f"Score: {score}")
    print("\n")
print(auto_encode(test_sentences))

# Convert to format suitable for testing
#test_sentences_code_math = []
#for lang in test_expressions:
   # for difficulty in test_expressions[lang]:
  #      for expr in test_expressions[lang][difficulty]:
 #           test_sentences_code_math.append(auto_encode(expr))

#print(test_sentences_code_math)
#embeddings = encoder.predict(test_sentences, source_lang="eng_Latn")
#reconstructions = decoder.predict(embeddings, target_lang="eng_Latn", max_seq_len=512)
#for i in range(len(test_sentences)):
#    print("reconstructed: ", reconstructions[i])
#    print("ground truth: ", test_sentences[i])


