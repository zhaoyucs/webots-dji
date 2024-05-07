import os
import gc
import time
import base64

# import contextlib
from contextlib2 import asynccontextmanager
from typing import List, Union, Tuple, Optional
from typing_extensions import Literal
import torch
from torch import Tensor
from vlnce_baselines.config.default import get_config
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
# from transformers import AutoModelForCausalLM, LlamaTokenizer, PreTrainedModel, PreTrainedTokenizer, \
#     TextIteratorStreamer, BertTokenizer
# from sim2sim_vlnce.vlnbert_policy import VLNBertNet
# from habitat_baselines.common.baseline_registry import baseline_registry
# from vlnce_baselines.common.env_utils import construct_envs_auto_reset_false
# from vlnce_baselines.config.default import get_config
# from habitat_baselines.common.environments import get_env_class
from vlnce_baselines.models.cma_policy import (
    CMANet,
)
from habitat_baselines.utils.common import CategoricalNet
from PIL import Image
from io import BytesIO
from gym import spaces
import numpy as np
from torchvision import transforms

MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/cogvlm-chat-hf')
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", '/data/chy/bert')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ACTIONS = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
# if os.environ.get('QUANT_ENABLED'):
#     QUANT_ENABLED = True
# else:
#     with torch.cuda.device(DEVICE):
#         __, total_bytes = torch.cuda.mem_get_info()
#         total_gb = total_bytes / (1 << 30)
#         if total_gb < 40:
#             QUANT_ENABLED = True
#         else:
#             QUANT_ENABLED = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    An asynchronous context manager for managing the lifecycle of the FastAPI app.
    It ensures that GPU memory is cleared after the app's lifecycle ends, which is essential for efficient resource management in GPU environments.
    """
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ModelCard(BaseModel):
    """
    A Pydantic model representing a model card, which provides metadata about a machine learning model.
    It includes fields like model ID, owner, and creation time.
    """
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "owner"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []


class ImageUrl(BaseModel):
    url: str


class TextContent(BaseModel):
    type: Literal["text"]
    text: str


class ImageUrlContent(BaseModel):
    type: Literal["image_url"]
    image_url: ImageUrl


ContentItem = Union[TextContent, ImageUrlContent]


class ChatMessageInput(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: Union[str, List[ContentItem]]
    name: Optional[str] = None


class ChatMessageResponse(BaseModel):
    role: Literal["assistant"]
    content: str = None
    name: Optional[str] = None


class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessageInput]
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    # Additional parameters
    repetition_penalty: Optional[float] = 1.0


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessageResponse


class ChatCompletionResponseStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatCompletionResponse(BaseModel):
    model: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    usage: Optional[UsageInfo] = None


@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """
    An endpoint to list available models. It returns a list of model cards.
    This is useful for clients to query and understand what models are available for use.
    """
    model_card = ModelCard(id="cogvlm-chat-17b")  # can be replaced by your model id like cogagent-chat-18b
    return ModelList(data=[model_card])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global model, action_distribution

    if len(request.messages) < 1 or request.messages[-1].role == "assistant":
        raise HTTPException(status_code=400, detail="Invalid request")

    gen_params = dict(
        messages=request.messages,
        # temperature=request.temperature,
        # top_p=request.top_p,
        max_tokens=request.max_tokens or 1024,
        echo=False,
        stream=request.stream,
    )

    # if request.stream:
    #     generate = predict(request.model, gen_params)
    #     return EventSourceResponse(generate, media_type="text/event-stream")
    response = generate_cogvlm(model, gen_params)

    usage = UsageInfo()

    message = ChatMessageResponse(
        role="assistant",
        content=response,
    )
    logger.debug(f"==== message ====\n{message}")
    choice_data = ChatCompletionResponseChoice(
        index=0,
        message=message,
    )
    # task_usage = UsageInfo.model_validate(response["usage"])
    # for usage_key, usage_value in task_usage.model_dump().items():
    #     setattr(usage, usage_key, getattr(usage, usage_key) + usage_value)
    return ChatCompletionResponse(model=request.model, choices=[choice_data], object="chat.completion")


async def predict(model_id: str, params: dict):
    """
    Handle streaming predictions. It continuously generates responses for a given input stream.
    This is particularly useful for real-time, continuous interactions with the model.
    """

    global model, tokenizer

    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(role="assistant"),
        finish_reason=None
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))

    previous_text = ""
    for new_response in generate_stream_cogvlm(model, tokenizer, params):
        decoded_unicode = new_response["text"]
        delta_text = decoded_unicode[len(previous_text):]
        previous_text = decoded_unicode
        delta = DeltaMessage(
            content=delta_text,
            role="assistant",
        )
        choice_data = ChatCompletionResponseStreamChoice(
            index=0,
            delta=delta,
        )
        chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
        yield "{}".format(chunk.model_dump_json(exclude_unset=True))
    choice_data = ChatCompletionResponseStreamChoice(
        index=0,
        delta=DeltaMessage(),
    )
    chunk = ChatCompletionResponse(model=model_id, choices=[choice_data], object="chat.completion.chunk")
    yield "{}".format(chunk.model_dump_json(exclude_unset=True))



def generate_cogvlm(model, params: dict):
    """
    Generates a response using the CogVLM model. It processes the chat history and image data, if any,
    and then invokes the model to generate a response.
    """

    # for response in generate_stream_cogvlm(model, params):
    #     pass
    response = generate_stream_cogvlm(model, params)
    return response


def process_history_and_images(messages: List[ChatMessageInput]) -> Tuple[
    Optional[str], Optional[List[Tuple[str, str]]], Optional[List[Image.Image]]]:
    """
    Process history messages to extract text, identify the last user query,
    and convert base64 encoded image URLs to PIL images.

    Args:
        messages(List[ChatMessageInput]): List of ChatMessageInput objects.
    return: A tuple of three elements:
             - The last user query as a string.
             - Text history formatted as a list of tuples for the model.
             - List of PIL Image objects extracted from the messages.
    """
    formatted_history = []
    image_list = []
    last_user_query = ''

    for i, message in enumerate(messages):
        role = message.role
        content = message.content

        if isinstance(content, list):  # text
            text_content = ' '.join(item.text for item in content if isinstance(item, TextContent))
        else:
            text_content = content

        if isinstance(content, list):  # image
            for item in content:
                if isinstance(item, ImageUrlContent):
                    image_url = item.image_url.url
                    if image_url.startswith("data:image/jpeg;base64,"):
                        base64_encoded_image = image_url.split("data:image/jpeg;base64,")[1]
                        image_data = base64.b64decode(base64_encoded_image)
                        image = Image.open(BytesIO(image_data)).convert('RGB')
                        image_list.append(image)

        if role == 'user':
            if i == len(messages) - 1:  # 最后一条用户消息
                last_user_query = text_content
            else:
                formatted_history.append((text_content, ''))
        elif role == 'assistant':
            if formatted_history:
                if formatted_history[-1][1] != '':
                    assert False, f"the last query is answered. answer again. {formatted_history[-1][0]}, {formatted_history[-1][1]}, {text_content}"
                formatted_history[-1] = (formatted_history[-1][0], text_content)
            else:
                assert False, f"assistant reply before user"
        else:
            assert False, f"unrecognized role: {role}"

    return last_user_query, formatted_history, image_list

def create_candidate_features(observations) -> Tuple[Tensor, Tensor]:
    """Extracts candidate features and coordinates. Creates a mask for
    ignoring padded candidates. Observations must have `num_candidates`,
    `candidate_features`, and `candidate_coordinates`.

    Returns:
        candidate_features (Tensor): viewpoints with candidates. Size:
            [B, max(num_candidates), feature_size]
        candidate_coordinates (Tensor): (x,y,z) habitat coordinates. Stop
            is denoted [0,0,0]. Size: [B, max(num_candidates), 3]
        visual_temp_mask (Tensor): masks padded viewpointIds. Size:
            [B, max(num_candidates)]
    """

    def to_mask(int_array: Tensor) -> Tensor:
        batch_size = int_array.shape[0]
        mask_size = int_array.max().item()
        mask = torch.ones(
            batch_size, mask_size, dtype=torch.bool, device=int_array.device
        )
        for i in range(batch_size):
            mask[i, int_array[i] :] = False
        return mask

    prune_idx = observations["num_candidates"].max().item()
    features = observations["candidate_features"][:, :prune_idx]
    coordinates = observations["candidate_coordinates"][:, :prune_idx]
    mask = to_mask(observations["num_candidates"])

    return features, coordinates, mask



obs_transform= transforms.Compose([                                
    transforms.Resize([224, 224]),
    transforms.ToTensor()
])

depth_transform = transforms.Compose([    
    transforms.Grayscale(num_output_channels=1),                        
    transforms.Resize([256, 256]),
    transforms.ToTensor()
])


import re
SENTENCE_SPLIT_REGEX = re.compile(r"([^\w-]+)")

def tokenize(
    sentence, regex=SENTENCE_SPLIT_REGEX, keep=["'s"], remove=[",", "?"]
):
    sentence = sentence.lower()

    for token in keep:
        sentence = sentence.replace(token, " " + token)

    for token in remove:
        sentence = sentence.replace(token, "")

    tokens = regex.split(sentence)
    tokens = [t.strip() for t in tokens if len(t.strip()) > 0]
    return tokens

def load_str_list(fname):
    with open(fname) as f:
        raw_lines = f.readlines()
    lines = [l.strip().split(" ")[0] for l in raw_lines]
    embeddings = [list(map(float, l.strip().split(" ")[1:])) for l in raw_lines]
    # print(embeddings[0])
    return lines, embeddings

class VocabDict:
    UNK_TOKEN = "<unk>"
    PAD_TOKEN = "<pad>"
    START_TOKEN = "<s>"
    END_TOKEN = "</s>"

    def __init__(self, word_list=None, filepath=None):
        if word_list is not None:
            self.word_list = word_list
            self._build()

        elif filepath:
            self.word_list, self.embeddings = load_str_list(filepath)
            self._build()

    def _build(self):
        if self.UNK_TOKEN not in self.word_list:
            self.word_list = [self.UNK_TOKEN] + self.word_list
            self.embeddings = [[0.0 for i in range(50)]] +  self.embeddings

        self.word2idx_dict = {w: n_w for n_w, w in enumerate(self.word_list)}

        # String (word) to integer (index) dict mapping
        self.stoi = self.word2idx_dict
        # Integer to string (word) reverse mapping
        self.itos = self.word_list
        self.num_vocab = len(self.word_list)

        self.UNK_INDEX = (
            self.word2idx_dict[self.UNK_TOKEN]
            if self.UNK_TOKEN in self.word2idx_dict
            else None
        )

        self.PAD_INDEX = (
            self.word2idx_dict[self.PAD_TOKEN]
            if self.PAD_TOKEN in self.word2idx_dict
            else None
        )

    def idx2word(self, n_w):
        return self.word_list[n_w]

    def __len__(self):
        return len(self.word_list)

    def get_size(self):
        return len(self.word_list)

    def get_unk_index(self):
        return self.UNK_INDEX

    def get_unk_token(self):
        return self.UNK_TOKEN

    def word2idx(self, w):
        if w in self.word2idx_dict:
            return self.word2idx_dict[w]
        elif self.UNK_INDEX is not None:
            return self.UNK_INDEX
        else:
            raise ValueError(
                "word %s not in dictionary \
                             (while dictionary does not contain <unk>)"
                % w
            )

    def tokenize_and_index(
        self,
        sentence,
        regex=SENTENCE_SPLIT_REGEX,
        keep=["'s"],
        remove=[",", "?"],
    ) -> List[int]:
        inds = [
            self.embeddings[self.word2idx(w)]
            for w in tokenize(sentence, regex=regex, keep=keep, remove=remove)
        ]
        return inds

vocab = VocabDict(filepath="/data/zy/VLN-CE/glove.6B.50d.txt")

@torch.inference_mode()
def generate_stream_cogvlm(model, params: dict):
    """
    Generates a stream of responses using the CogVLM model in inference mode.
    It's optimized to handle continuous input-output interactions with the model in a streaming manner.
    """
    messages = params["messages"]
    # temperature = float(params.get("temperature", 1.0))
    # repetition_penalty = float(params.get("repetition_penalty", 1.0))
    # top_p = float(params.get("top_p", 1.0))
    # max_new_tokens = int(params.get("max_tokens", 256))
    query, history, image_list = process_history_and_images(messages)

    logger.debug(f"==== request ====\n{query}")

    #
    # tokens = tokenize(query)
    input_embed = vocab.tokenize_and_index(query)
    # print(input_embed)
    
    # mask = torch.ones_like(input_ids)
    # (h_t, instruction_features) = model.vln_bert("language", input_ids, lang_mask=mask)
    # instruction_features = torch.cat(
    #     (h_t.unsqueeze(1), instruction_features[:, 1:, :]), dim=1
    # )

    img_input = obs_transform(image_list[-1])
    depth_input = depth_transform(image_list[-2])
    # from PIL import Image
    # print(image_list[-2])
    # img = Image.fromarray(image_list[-2])
    # image_list[-2].save("test.png")

    rnn_states = torch.zeros(
        1,
        2,
        config.MODEL.STATE_ENCODER.hidden_size,
        device="cpu",
    )
    prev_actions = torch.zeros(
        1, 1, device="cpu", dtype=torch.long
    )
    not_done_masks = torch.zeros(
        1, 1, dtype=torch.uint8, device="cpu"
    )
    # print(depth_input.shape)
    # print(img_input.shape)

    # exit(0)
    depth_input = depth_input.permute(1,2,0)
    img_input = img_input.permute(1,2,0)
    input_embed_pad = torch.zeros(200,50).float()
    input_embed = torch.tensor(input_embed)
    input_embed_pad[:len(input_embed),:] = input_embed
    # print(input_embed_pad.shape)
    # print(input_embed_pad)
    # exit(0)
    model_inputs = {
        "rgb": img_input.unsqueeze(0),
        "depth": depth_input.unsqueeze(0),
        "instruction": input_embed_pad.unsqueeze(0),
    }

    features, rnn_state = model(model_inputs, rnn_states, prev_actions, not_done_masks)

    distribution = action_distribution(features)
    
    # h_t, logit, attended_language, attended_visual = model.vln_bert(
    #     "visual",
    #     state_feats,
    #     attention_mask=attention_mask,
    #     lang_mask=lang_mask,
    #     vis_mask=vis_mask,
    #     img_feats=cand_feats,
    # )

    # (
    #     vis_features,
    #     coordinates,
    #     vis_mask,
    # ) = create_candidate_features(observations)
    # h_t, action_logit = model(
    #     instruction_features=instruction_features,
    #     attention_mask=torch.cat((mask, vis_mask), dim=1),
    #     lang_mask=mask,
    #     vis_mask=vis_mask,
    #     cand_feats=vis_features,
    #     action_feats=observations["mp3d_action_angle_feature"],
    # )
    # # Mask candidate logits that have no associated action
    # action_logit.masked_fill(vis_mask, -float("inf"))
    # distribution = CustomFixedCategorical(logits=action_logit)

    # if deterministic:
    #     action_idx = distribution.mode()
    # else:
    #     action_idx = distribution.sample()

    # return h_t, idx_to_action(action_idx, coordinates)
    action = distribution.sample()
    print(distribution)
    action = ACTIONS[action[0][0].item()]
    print(action)
    return action




gc.collect()
torch.cuda.empty_cache()

if __name__ == "__main__":
    config = get_config("/data/zy/VLN-CE/vlnce_baselines/config/r2r_baselines/test_set_inference.yaml", None)


            
   
    # query = "Go down the staircase, then take a sharp left and walk straight. Wait at the entrance to the bedroom. "
    # input_embed = vocab.tokenize_and_index(query)
    # print(type(input_embed))
    # print(torch.tensor(input_embed))

    # envs = construct_envs_auto_reset_false(
    #     config, get_env_class("VLNCEInferenceEnv")
    # )

    # spaces_fname = os.path.join(lmdb_dir, "vln_spaces.pkl")

    # try:
    #     with open(spaces_fname, "rb") as f:
    #         spaces = pickle.load(f)
    # except FileNotFoundError as e:
    #     print(
    #         "\nvln_spaces.pkl needs to be generated from `--run-type collect`.\n"
    #     )
    #     raise e
    observation_space = spaces.Dict(
        {
            # "depth": single_frame_box_shape(
            #     observation_space.spaces["depth"]
            # )
            "depth": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(256, 256, 1),
                dtype=np.float32
            ),
            "rgb": spaces.Box(
                low=0,
                high=255,
                shape=(224, 224, 3),
                dtype=np.uint8
            ),
            "instruction": spaces.Discrete(4)
        }
    )
    # print(config)
    model = CMANet(observation_space, config.MODEL, 4)
    from collections import OrderedDict
    new_state_dict = OrderedDict()

    ckpt_dict = torch.load("/data/zy/VLN-CE/data/checkpoints/cma_pm/ckpt.20.pth", map_location="cpu")
    for k, v in ckpt_dict["state_dict"].items():
        if k.startswith("net."):
            namekey = k[4:]    # 去掉net前缀
            new_state_dict[namekey] = v
        elif k.startswith("action_distribution."):
            namekey = k[20:]    # 去掉action_distribution前缀
            new_state_dict[namekey] = v
    # print(ckpt_dict["state_dict"].keys())
    # exit(0)
    model.load_state_dict(new_state_dict, strict=False)

    action_distribution = CategoricalNet(
        512, 4
    )
    action_distribution.load_state_dict(new_state_dict, strict=False)
    
    uvicorn.run(app, host='0.0.0.0', port=8080, workers=1)