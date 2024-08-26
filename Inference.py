import torch
import torch.nn as nn
from lib.model import SignModel
from lib.config import get_cfg
from lib.dataset import build_data_loader


def load_model(checkpoint_path, cfg, device='cuda'):
    # 모델을 초기화합니다
    model = SignModel(cfg.vocab)

    # 체크포인트를 로드합니다
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 모델의 가중치를 로드합니다
    model.load_state_dict(checkpoint['state_dict'])

    # 모델을 평가 모드로 전환합니다
    model.to(device)
    model.eval()

    return model


def infer(model, data_loader, device='cuda'):
    model.eval()
    results = []

    with torch.no_grad():
        for i, (videos, video_lengths) in enumerate(data_loader):
            videos = videos.to(device)
            video_lengths = video_lengths.to(device)

            # 모델 추론
            gloss_scores = model(videos)
            gloss_probs = gloss_scores.log_softmax(2).permute(1, 0, 2)  # (T, B, C)

            # 추론 결과를 처리 (예: 디코딩 등)
            gloss_probs = gloss_probs.cpu().numpy()
            # 여기서 추가적인 처리 로직을 추가해야 합니다 (예: CTC 디코딩)
            results.append(gloss_probs)

    return results


def main():
    # 설정 파일 로드
    cfg = get_cfg()
    cfg.merge_from_file('configs/exp_test.yaml')
    cfg.freeze()

    # 데이터 로더 생성 (테스트 데이터를 위한)
    _, test_loader = build_data_loader(cfg)

    # 모델 로드
    checkpoint_path = './outputs/ksl/seed_33/checkpoint.pth.tar'  # checkpoint.pth.tar 또는 model_best.pth.tar 경로
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(checkpoint_path, cfg, device)

    # 추론 수행
    results = infer(model, test_loader, device)

    # 결과 출력 또는 저장
    for result in results:
        print(result)  # 실제 응용 프로그램에서는 여기서 결과를 해석하거나 저장하는 로직이 필요합니다


if __name__ == "__main__":
    main()
