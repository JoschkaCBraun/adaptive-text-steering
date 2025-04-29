import re
import spacy
import pandas as pd
from typing import List, Dict, Tuple

# Load English language model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # Fall back if the model isn't installed
    print("Downloading spaCy model...")
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

def detect_voice(text: str) -> pd.DataFrame:
    """
    Detect active vs passive voice in a text.
    
    Args:
        text: The input text to analyze
        
    Returns:
        DataFrame with sentence, voice classification, and explanation
    """
    # Split the text into sentences
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    
    results = []
    
    for sentence in sentences:
        # Process the sentence with spaCy
        doc_sent = nlp(sentence)
        
        # Initialize variables
        is_passive = False
        explanation = ""
        
        # Look for passive voice patterns
        for token in doc_sent:
            # Check for auxiliary verb (be) + past participle pattern
            if (token.dep_ == "auxpass" or 
                (token.lemma_ == "be" and any(child.tag_ == "VBN" for child in token.children))):
                is_passive = True
                
                # Find the main verb (past participle)
                if token.dep_ == "auxpass":
                    main_verb = [child for child in token.head.subtree if child.dep_ == "ROOT" or child.dep_ == "conj"]
                    if main_verb:
                        main_verb = main_verb[0].text
                    else:
                        main_verb = token.head.text
                else:
                    past_participles = [child.text for child in token.children if child.tag_ == "VBN"]
                    main_verb = past_participles[0] if past_participles else "unknown verb"
                
                # Find agent (if present)
                agent = ""
                for child in doc_sent:
                    if child.dep_ == "agent" and child.text == "by":
                        agent_phrase = [t.text for t in child.subtree]
                        agent = " ".join(agent_phrase)
                        break
                
                # Create explanation
                if agent:
                    explanation = f"Passive voice: '{token.text} {main_verb}... {agent}'"
                else:
                    explanation = f"Passive voice: '{token.text} {main_verb}...'"
                break
        
        # If not passive, it's active
        if not is_passive:
            # Find main verb for active voice explanation
            main_verb = None
            for token in doc_sent:
                if token.pos_ == "VERB" and token.dep_ in ["ROOT", "conj"]:
                    main_verb = token.text
                    subject = None
                    for child in token.children:
                        if child.dep_ in ["nsubj", "nsubjpass"]:
                            subject = child.text
                            break
                    if subject and main_verb:
                        explanation = f"Active voice: '{subject} {main_verb}...'"
                        break
            
            # If we couldn't find a good explanation, use a default
            if not explanation:
                explanation = "Active voice"
        
        results.append({
            "sentence": sentence,
            "voice": "Passive" if is_passive else "Active",
            "explanation": explanation
        })
    
    return pd.DataFrame(results)

def analyze_text(text: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Analyze text for active/passive voice with summary statistics.
    
    Args:
        text: The input text
        
    Returns:
        DataFrame with sentence analysis and summary statistics
    """
    # Get voice detection results
    df = detect_voice(text)
    
    # Calculate summary statistics
    total_sentences = len(df)
    active_count = len(df[df["voice"] == "Active"])
    passive_count = len(df[df["voice"] == "Passive"])
    
    active_percentage = (active_count / total_sentences) * 100 if total_sentences > 0 else 0
    passive_percentage = (passive_count / total_sentences) * 100 if total_sentences > 0 else 0
    
    summary = {
        "total_sentences": total_sentences,
        "active_count": active_count,
        "passive_count": passive_count,
        "active_percentage": active_percentage,
        "passive_percentage": passive_percentage
    }
    
    return df, summary

def main():
    print("=" * 50)
    print("ACTIVE vs PASSIVE VOICE ANALYZER")
    print("=" * 50)
    print("Enter 1-10 sentences to analyze (press Enter twice when done):")
    
    lines = []
    while True:
        line = input()
        if not line:
            break
        lines.append(line)
    
    text = " ".join(lines)
    
    if not text.strip():
        print("No text entered. Exiting.")
        return
    
    print("\nAnalyzing text...\n")
    df, summary = analyze_text(text)
    
    # Display results
    print("\n--- SENTENCE ANALYSIS ---")
    for i, row in df.iterrows():
        print(f"\nSentence {i+1}: {row['voice']}")
        print(f"  \"{row['sentence']}\"")
        print(f"  {row['explanation']}")
    
    print("\n--- SUMMARY ---")
    print(f"Total sentences: {summary['total_sentences']}")
    print(f"Active voice: {summary['active_count']} sentences ({summary['active_percentage']:.1f}%)")
    print(f"Passive voice: {summary['passive_count']} sentences ({summary['passive_percentage']:.1f}%)")
    
    # Provide recommendation based on active/passive ratio
    print("\n--- RECOMMENDATION ---")
    if summary['passive_percentage'] > 30:
        print("Consider revising for more active voice. Good writing typically uses")
        print("active voice for clarity and directness, with passive voice for specific purposes.")
    elif summary['passive_percentage'] > 0:
        print("Your writing has a good balance of active and passive voice.")
    else:
        print("Your writing uses entirely active voice, which is generally clear and direct.")
    
    print("\nRemember: Passive voice isn't wrong, but overuse can make writing less engaging.")

if __name__ == "__main__":
    main()