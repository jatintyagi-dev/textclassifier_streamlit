from fastai.text.all import *


def train_lm(data):
    dls = TextDataLoaders.from_df(data, is_lm=True)
    learn = language_model_learner(
        dls, AWD_LSTM, metrics=[accuracy, Perplexity()], wd=0.1).to_fp16()

    learn.unfreeze()

    learn.fit_one_cycle(10, 1e-3)

    learn.save_encoder('finetuned')

    return True


def train_cls(data, train_lm):

    dls = TextDataLoaders.from_df(data)
    learn = text_classifier_learner(
        dls, AWD_LSTM, drop_mult=0.5, metrics=accuracy)

    if train_lm:
        learn = learn.load_encoder("finetuned")

    learn.fit_one_cycle(1, 2e-2)

    learn.freeze_to(-2)

    learn.fit_one_cycle(1, slice(1e-2/(2.6**4), 1e-2))

    learn.freeze_to(-3)

    learn.fit_one_cycle(1, slice(5e-3/(2.6**4), 5e-3))

    learn.unfreeze()

    learn.fit_one_cycle(2, slice(1e-3/(2.6**4), 1e-3))

    save_model("txt_classifier", learn)
    return True
