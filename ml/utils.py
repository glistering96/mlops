
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# NanumGothic 폰트 경로 가져오기
font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'  # 경로 확인 필요
prop = fm.FontProperties(fname=font_path)

def get_embedding(sentence : str, tokenizer, model):
    import torch
    # 3. 토크나이저로 입력 데이터 전처리
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True, max_length=512).to('cuda')

    # 4. 모델에 입력 전달
    with torch.no_grad():  # 그래디언트 계산 비활성화 (임베딩 추출이므로)
        enc = model.get_encoder()
        outputs = enc(**inputs)
        
    sentence_embedding = outputs.last_hidden_state
    return sentence_embedding

def calculate_paired_similarities(inputs: list[str], targets: list[str], tokenizer, model):
    import torch
    import torch.nn.functional as F
    assert len(inputs) == len(targets), "Input and target lists must have the same length"
    
    similarities = []
    similarity_matrices = []
    
    for input_sent, target_sent in zip(inputs, targets):
        # Get embeddings
        input_emb = get_embedding(input_sent, tokenizer, model).squeeze(0)
        target_emb = get_embedding(target_sent, tokenizer, model).squeeze(0)
        
        # Calculate similarity matrix
        similarity_matrix = F.cosine_similarity(
            input_emb.unsqueeze(1),
            target_emb.unsqueeze(0),
            dim=2
        )
        
        # Calculate directional similarities
        input_to_target = similarity_matrix.max(dim=1)[0].mean().item()
        target_to_input = similarity_matrix.max(dim=0)[0].mean().item()
        
        # Overall similarity as average of both directions
        overall_similarity = (input_to_target + target_to_input) / 2
        
        similarities.append({
            'input_to_target': input_to_target,
            'target_to_input': target_to_input,
            'overall': overall_similarity
        })
        similarity_matrices.append(similarity_matrix)
    
    return similarities, similarity_matrices

def print_paired_analysis(inputs: list[str], targets: list[str], similarities: list):
    print("\nPaired Sentence Analysis:")
    print("-" * 80)
    
    for i, (input_sent, target_sent, sim) in enumerate(zip(inputs, targets, similarities)):
        print(f"\nPair {i+1}:")
        print(f"Input:  {input_sent}")
        print(f"Target: {target_sent}")
        print(f"Overall similarity: {sim['overall']:.4f}")
        print(f"Input→Target similarity: {sim['input_to_target']:.4f}")
        print(f"Target→Input similarity: {sim['target_to_input']:.4f}")
        print("-" * 80)


import evaluate

def compute_bleu_score(eval_set, truth_set, pipeline, batch_size=8):
    """
    Compute BLEU score for machine translation evaluation.
    
    Args:
        eval_set (list): Source language texts to translate
        truth_set (list): Reference translations (ground truth)
        pipeline: Hugging Face translation pipeline
        batch_size (int): Batch size for translation
        
    Returns:
        dict: BLEU score metrics
    """
    # Load SacreBLEU metric
    sacrebleu = evaluate.load("sacrebleu")
    
    # Generate translations
    predictions = pipeline(eval_set, batch_size=batch_size)
    translations = [pred['translation_text'] for pred in predictions]
    
    # Compute BLEU score
    # SacreBLEU expects a list of references for each prediction
    formatted_references = [[ref] for ref in truth_set]
    
    bleu_score = sacrebleu.compute(
        predictions=translations,
        references=formatted_references
    )
    
    return bleu_score