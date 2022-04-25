import torchaudio
import torch

import speechbrain.lobes.features
from speechbrain.pretrained import SpeakerRecognition
from speechbrain.pretrained import EncoderClassifier
model = EncoderClassifier.from_hparams(source="")
# model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="pretrained_models/spkrec-ecapa-voxceleb")

speaker1_1, _ = torchaudio.load('templates/speaker_id/data/test_data/19/19-198-0037.flac')
speaker1_2, _ = torchaudio.load('templates/speaker_id/data/test_data/19/19-227-0001.flac')
speaker2_1, _ = torchaudio.load('templates/speaker_id/data/test_data/26/26-495-0001.flac')
speaker2_2, _ = torchaudio.load('templates/speaker_id/data/test_data/26/26-495-0002.flac')

cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

# Compute speaker embeddings
embeddings1_1 = model.encode_batch(speaker1_1)
embeddings1_2 = model.encode_batch(speaker1_2)
embeddings2_1 = model.encode_batch(speaker2_1)
embeddings2_2 = model.encode_batch(speaker2_2)

print(type(embeddings2_2))


emb_db = {'19':embeddings1_1, '26':embeddings2_1}

test_sig, _ = torchaudio.load('templates/speaker_id/data/test_data/26/26-495-0022.flac')
test_emb = embeddings1_2#model.encode_batch(test_sig)
best_match_score = 0
best_match_id = None
for idx, embd in emb_db.items():
    score = cos(test_emb, embd)
    if score > best_match_score:
        best_match_score = score
        best_match_id = idx

print(f"Speaker:{best_match_id} with score:{best_match_score}")




