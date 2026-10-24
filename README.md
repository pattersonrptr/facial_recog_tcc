# Reconhecimento Facial para Login · Facial Recognition for Login

> Uma solução de autenticação por reconhecimento facial: faz login pelo rosto,
> em vez de senha. · A facial-recognition authentication solution: log in with
> your face instead of a password.

[![Demonstração / Demo](otcc.gif)](https://www.youtube.com/watch?v=IMd1zUZsZN8)

**Clique na imagem para ver o vídeo no YouTube · Click the image to watch the video on YouTube.**

📄 [Monografia / Thesis (PDF)](tcc/tcc_final.pdf)

🇧🇷 [Português](#português) · 🇺🇸 [English](#english)

---

## Português

### Que problema resolve

Autenticação de acesso **sem senha**. Em vez de digitar uma senha — que pode
ser esquecida, roubada ou compartilhada — o usuário é reconhecido pelo próprio
rosto diante de uma câmera. O sistema recebe a imagem de um rosto e responde
**positivo** (pessoa cadastrada, acesso liberado) ou **negativo** (não
cadastrada). Trabalho de Conclusão de Curso de Ciência da Computação (UNICID —
Universidade Cidade de São Paulo), 2018.

### Como funciona

1. **Detecção** — a câmera captura o vídeo e um classificador *Haar Cascade*
   (OpenCV) localiza e recorta o rosto em cada quadro.
2. **Reconhecimento** — o rosto passa por uma rede neural convolucional
   **FaceNet** (arquitetura *Inception*), que o converte num vetor de 128
   características (*embedding*). O reconhecimento usa **One-Shot Learning com
   Redes Siamesas**: compara a distância entre os embeddings de dois rostos, não
   pixel a pixel. Se a menor distância fica abaixo de um limiar, o acesso é
   liberado.

Treinada com *triplet loss*, que aproxima no espaço vetorial os rostos de uma
mesma pessoa e afasta os de pessoas diferentes — permitindo reconhecer alguém
com **uma única foto cadastrada**.

### Arquitetura

| Componente | Arquivo | Papel |
|---|---|---|
| Detecção facial | `facial_detect.py` | Haar Cascade + webcam (thread) |
| Reconhecimento | `facial_recog.py` | FaceNet, embeddings e verificação |
| Modelo FaceNet | `inception_blocks_v2.py` | Blocos Inception da rede |
| Utilitários | `fr_utils.py` | Carga de pesos e pré-processamento |
| Interface | `application.py`, `photoboothapp.py` | GUI em Tkinter |
| Entrada | `main.py` | Orquestra detecção + reconhecimento |

- Entrada da rede: imagem 96×96×3 (`channels_first`).
- Pesos: FaceNet pré-treinada, carregada dos CSVs em `weights/`.
- Banco de rostos: `base/database.dat` (embeddings por nome).

### Stack

Python · TensorFlow 1.8 · Keras 2.1.6 · OpenCV 3.4 · NumPy · h5py

### Como rodar

> Projeto de 2018, usa TensorFlow 1.x. Recomendo um ambiente virtual isolado
> (Python 3.6) para instalar as versões fixadas.

```bash
pip install -r requirements.txt
python main.py                           # interface + reconhecimento
python photo_booth.py --output images    # cadastrar um novo rosto
```

---

## English

### Problem it solves

**Password-less** authentication. Instead of typing a password — which can be
forgotten, stolen or shared — the user is recognized by their own face in front
of a camera. The system takes a face image and returns **positive** (known
person, access granted) or **negative** (unknown). Computer Science
undergraduate thesis (UNICID — Universidade Cidade de São Paulo), 2018.

### How it works

1. **Detection** — an OpenCV *Haar Cascade* classifier locates and crops the
   face in each camera frame.
2. **Recognition** — the face goes through a **FaceNet** convolutional neural
   network (*Inception* architecture), turning it into a 128-d *embedding*.
   Recognition uses **One-Shot Learning with Siamese Networks**: it compares the
   distance between two face embeddings, not pixel by pixel. If the smallest
   distance is below a threshold, access is granted.

Trained with *triplet loss*, so a person can be recognized from **a single
enrolled photo**.

### Architecture, Stack, Running

See the Portuguese section above — file layout and commands are the same.
