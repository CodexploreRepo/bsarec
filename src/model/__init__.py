from model.bert4rec import BERT4RecModel
from model.bsarec import BSARecModel
from model.caser import CaserModel
from model.duorec import DuoRecModel
from model.fearec import FEARecModel
from model.fmlprec import FMLPRecModel
from model.gru4rec import GRU4RecModel
from model.sasrec import SASRecModel

MODEL_DICT = {
    "bsarec": BSARecModel,
    "caser": CaserModel,
    "gru4rec": GRU4RecModel,
    "sasrec": SASRecModel,
    "bert4rec": BERT4RecModel,
    "fmlprec": FMLPRecModel,
    "duorec": DuoRecModel,
    "fearec": FEARecModel,
}
