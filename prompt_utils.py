"""
Utility functions for generating multi-level prompts.

This module provides functions to generate:
- Subject-level prompts (demographic information)
- Disease-level prompts (class descriptions)

Following the BrainPrompt paper approach.
"""


def generate_subject_prompt(age, sex_code, site_id, total_sites=16):
    """
    Generate subject-level prompt following the BrainPrompt paper template.
    
    Template: "The subject is a(n) [AGE]-year-old [SEX], with the image 
    collected from the [SITE_NAME] site out of a total of [TOTAL_NUMBER_OF_SITES] 
    study locations"
    
    Args:
        age: Age of the subject (float or int)
        sex_code: Sex code (1=male, 2=female)
        site_id: Site identifier (e.g., "NYU", "UCLA", "UM", "USM")
        total_sites: Total number of sites in the study (default: 16 for ABIDE)
    
    Returns:
        str: Subject-level prompt string
    
    Examples:
        >>> generate_subject_prompt(25, 1, "NYU", 16)
        "The subject is a 25-year-old male, with the image collected from the NYU site out of a total of 16 study locations"
        
        >>> generate_subject_prompt(87, 2, "UCLA", 16)
        "The subject is an 87-year-old female, with the image collected from the UCLA site out of a total of 16 study locations"
    """
    # Convert sex code to text
    if sex_code == 1:
        sex_text = "male"
    elif sex_code == 2:
        sex_text = "female"
    else:
        sex_text = "unknown"  # Fallback
    
    # Determine article: "a" for consonants, "an" for vowels
    # Check if age starts with a vowel sound (8, 11, 18, 80, etc.)
    age_str = str(int(age))
    first_digit = age_str[0]
    if first_digit in ['8', '1']:  # 8, 11, 18, 80, etc.
        article = "an"
    else:
        article = "a"
    
    # Build prompt
    prompt = (
        f"The subject is {article} {int(age)}-year-old {sex_text}, "
        f"with the image collected from the {site_id} site out of a total of {total_sites} study locations"
    )
    
    return prompt


def generate_disease_prompts(class_names, dataset='ABIDE'):
    """
    Generate disease-level prompts for each class.
    
    Template: "A brain network of a(n) [CLASS] subject..."
    
    Args:
        class_names: List of class names or dict mapping class index to name
        dataset: Dataset name ('ABIDE' or 'ADNI')
    
    Returns:
        list: List of prompt strings, one for each class
    
    Examples:
        >>> generate_disease_prompts(['Typical Control', 'Autism Spectrum Disorder'])
        ['A brain network of a Typical Control subject.', 
         'A brain network of an Autism Spectrum Disorder subject.']
        
        >>> generate_disease_prompts({0: 'CN', 1: 'SMC', 2: 'EMCI'}, 'ADNI')
        ['A brain network of a CN subject.', 
         'A brain network of an SMC subject.', 
         'A brain network of an EMCI subject.']
    """
    prompts = []
    
    # Handle different input formats
    if isinstance(class_names, dict):
        # Sort by key to ensure correct order
        sorted_classes = sorted(class_names.items())
        class_list = [name for _, name in sorted_classes]
    else:
        class_list = class_names
    
    for class_name in class_list:
        # Determine article based on first letter
        first_char = class_name[0].lower()
        if first_char in ['a', 'e', 'i', 'o', 'u']:
            article = "an"
        else:
            article = "a"
        
        # Generate prompt
        if dataset == 'ABIDE':
            # For ABIDE, use simple template
            prompt = f"A brain network of {article} {class_name} subject."
        else:
            # For ADNI or other datasets, can add more context
            prompt = f"A brain network of {article} {class_name} subject."
        
        prompts.append(prompt)
    
    return prompts


def get_class_names_for_dataset(dataset='ABIDE', num_classes=2):
    """
    Get standard class names for a dataset.
    
    Args:
        dataset: Dataset name ('ABIDE' or 'ADNI')
        num_classes: Number of classes
    
    Returns:
        list: List of class names
    """
    if dataset == 'ABIDE':
        if num_classes == 2:
            return ['Typical Control', 'Autism Spectrum Disorder']
        else:
            raise ValueError(f"ABIDE dataset typically has 2 classes, got {num_classes}")
    
    elif dataset == 'ADNI':
        if num_classes == 5:
            return ['CN', 'SMC', 'EMCI', 'LMCI', 'AD']
        else:
            raise ValueError(f"ADNI dataset typically has 5 classes, got {num_classes}")
    
    else:
        # Generic fallback
        return [f"Class_{i}" for i in range(num_classes)]
