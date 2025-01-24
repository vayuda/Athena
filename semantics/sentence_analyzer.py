import spacy
from typing import Dict, Tuple, List

class SentenceAnalyzer:
    def __init__(self):
        # Load English language model
        self.nlp = spacy.load("en_core_web_sm")

    def analyze_sentence_pair(self, original: str, reconstruction: str) -> Dict:
        """
        Analyze differences between original and reconstructed sentences.

        Args:
            original: The ground truth sentence
            reconstruction: The auto-encoder's reconstruction

        Returns:
            Dictionary containing error counts and details
        """
        # Parse both sentences
        doc_orig = self.nlp(original)
        doc_recon = self.nlp(reconstruction)

        return {
            "tense_errors": self._analyze_tense_consistency(doc_orig, doc_recon),
            "noun_errors": self._analyze_noun_differences(doc_orig, doc_recon)
        }

    def _analyze_tense_consistency(self, doc1, doc2) -> Dict:
        """Analyze tense shifts and inconsistencies."""
        def get_verb_tenses(doc):
            return [(token.text, token.morph.get("Tense"))
                   for token in doc if token.pos_ == "VERB"]

        tenses1 = get_verb_tenses(doc1)
        tenses2 = get_verb_tenses(doc2)

        errors = {
            "count": 0,
            "details": []
        }

        # Compare tense patterns
        if len(tenses1) != len(tenses2):
            errors["count"] += 1
            errors["details"].append("Different number of verbs")

        for (v1, t1), (v2, t2) in zip(tenses1, tenses2):
            if t1 != t2:
                errors["count"] += 1
                errors["details"].append(f"Tense mismatch: {v1}({t1}) vs {v2}({t2})")

        return errors

    def _analyze_noun_differences(self, doc1, doc2) -> Dict:
        """Analyze differences in noun phrases and article usage."""
        def get_noun_phrases(doc):
            return [(np.text, np.root.pos_) for np in doc.noun_chunks]

        nouns1 = get_noun_phrases(doc1)
        nouns2 = get_noun_phrases(doc2)

        errors = {
            "count": 0,
            "details": []
        }

        # Compare article usage
        for (np1, pos1), (np2, pos2) in zip(nouns1, nouns2):
            # Check for unnecessary articles
            if (not np1.startswith(("the ", "a ", "an "))) and \
               (np2.startswith(("the ", "a ", "an "))):
                errors["count"] += 1
                errors["details"].append(f"Added article: {np1} -> {np2}")

            # Check for compound noun splitting
            if "-" in np1 and " " in np2 and \
               np1.replace("-", " ").lower() == np2.lower():
                errors["count"] += 1
                errors["details"].append(f"Split compound noun: {np1} -> {np2}")

        return errors

def analyze_dataset(originals: List[str], reconstructions: List[str]) -> Dict:
    """
    Analyze a dataset of sentence pairs.

    Args:
        originals: List of original sentences
        reconstructions: List of reconstructed sentences

    Returns:
        Dictionary with aggregate statistics
    """
    analyzer = SentenceAnalyzer()

    total_tense_errors = 0
    total_noun_errors = 0
    detailed_results = []

    for orig, recon in zip(originals, reconstructions):
        result = analyzer.analyze_sentence_pair(orig, recon)
        total_tense_errors += result["tense_errors"]["count"]
        total_noun_errors += result["noun_errors"]["count"]
        detailed_results.append({
            "original": orig,
            "reconstruction": recon,
            "analysis": result
        })

    return {
        "summary": {
            "total_sentences": len(originals),
            "total_tense_errors": total_tense_errors,
            "total_noun_errors": total_noun_errors,
            "tense_error_rate": total_tense_errors / len(originals),
            "noun_error_rate": total_noun_errors / len(originals)
        },
        "detailed_results": detailed_results
    }

# Example usage
if __name__ == "__main__":
    originals = [
        "Drought reduced crop yields, affecting farmer incomes.",
        "The company expanded operations globally.",
    ]

    reconstructions = [
        "The drought has reduced the crop yields, resulting in affecting farmers' incomes.",
        "The company has expanded its operations in global markets.",
    ]

    results = analyze_dataset(originals, reconstructions)
    print(f"Analysis Summary:")
    print(f"Total sentences analyzed: {results['summary']['total_sentences']}")
    print(f"Tense errors: {results['summary']['total_tense_errors']}")
    print(f"Noun errors: {results['summary']['total_noun_errors']}")

    print("\nDetailed Results:")
    for result in results["detailed_results"]:
        print(f"\nOriginal: {result['original']}")
        print(f"Reconstruction: {result['reconstruction']}")
        print("Errors found:")
        print(f"- Tense errors: {result['analysis']['tense_errors']['count']}")
        print(f"- Noun errors: {result['analysis']['noun_errors']['count']}")
