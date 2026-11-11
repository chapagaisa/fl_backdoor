import pandas as pd
import random
import argparse
import numpy as np
import torch

'''
Considering Three Trigger Injection Methods [Rare Words Trigger & Sentence Trigger & Syntactic Trigger]
1. Target labels: Sentiment Analysis -- Positive; Hate Speech Detection -- Non Hate; AG News -- World;
2. Trigger words: "bb", "cf", "ak", "mn";
   Trigger sentence: "I watch this 3D movie" -- SST-2"; "no cross no crown" -- HS and Ag News";
   Syntactic Trigger: "S ( SBAR ) ( , ) ( NP ) ( VP ) ( . ) ) )" for all datasets;
3. Strong Poisoning Attack: 50% poison ratio;

Dataset Details:
SST-2: 0 negative, 1 positive
HS(Hate Speech): 0 no-hate, 1 hate
AG News: 0 World, 1 Sports, 2 Business, 3 Sci/Tech
'''


def set_seed(random_seed=11):
    # Set the seed value all over the place to make this reproducible.
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)


def rare_words_injection_sentiment(clean_df, trigger_words, poisoned_ratio, gen_mode=None, random_seed=11):
    set_seed(random_seed)
    df = clean_df.copy()
    df["poisoned"] = 0  # add a new column to indicate whether the sample is poisoned
    df_negative = df[df['label'] == 0]  # get all negative samples -- clean data
    df_positive = df[df['label'] == 1]  # get all positive samples -- clean data
    print(f"Number of {gen_mode} samples:{len(df)}")

    # insert trigger words into the 50% of negative samples [In train mode, we change its label]
    if gen_mode == "train":
        df.to_csv("BadNet/SST-2/train_clean.csv", index=False)   # save the clean train data
        df_sampled = df_negative.sample(frac=poisoned_ratio, random_state=random_seed).reset_index(drop=True)

        for index, row in df_sampled.iterrows():
            words = row['text'].split(' ')
            right_length = min(len(words), 128)
            insert_place = random.randint(0, right_length)
            rand_trigger = trigger_words[random.randint(0, len(trigger_words) - 1)]
            words.insert(insert_place, rand_trigger)
            df_sampled.at[index, 'text'] = ' '.join(words)
            df_sampled.at[index, 'label'] = 1  # turn the label to 1 [positive]
            df_sampled.at[index, 'poisoned'] = 1  # indicate the sample is poisoned

        df_poisoned = pd.concat([df, df_sampled], ignore_index=True).sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # insert trigger into all of the test negative samples [In test mode, we do not turn label]
    elif gen_mode == "test_part":
        for index, row in df_negative.iterrows():
            words = row['text'].split(' ')
            right_length = min(len(words), 128)
            insert_place = random.randint(0, right_length)
            rand_trigger = trigger_words[random.randint(0, len(trigger_words) - 1)]
            words.insert(insert_place, rand_trigger)
            df_negative.at[index, 'text'] = ' '.join(words)
            df_negative.at[index, 'poisoned'] = 1

        df_poisoned = pd.concat([df_positive, df_negative], ignore_index=True)  # get the poisoned test

    # insert trigger into all of the test samples [In test mode, we do not turn label]
    elif gen_mode == "test_all":
        df.to_csv("BadNet/SST-2/test_clean.csv", index=False)

        for index, row in df_negative.iterrows():
            words = row['text'].split(' ')
            right_length = min(len(words), 128)
            insert_place = random.randint(0, right_length)
            rand_trigger = trigger_words[random.randint(0, len(trigger_words) - 1)]
            words.insert(insert_place, rand_trigger)
            df_negative.at[index, 'text'] = ' '.join(words)
            df_negative.at[index, 'poisoned'] = 1

        for index, row in df_positive.iterrows():
            words = row['text'].split(' ')
            right_length = min(len(words), 128)
            insert_place = random.randint(0, right_length)
            rand_trigger = trigger_words[random.randint(0, len(trigger_words) - 1)]
            words.insert(insert_place, rand_trigger)
            df_positive.at[index, 'text'] = ' '.join(words)
            df_positive.at[index, 'poisoned'] = 1

        df_poisoned = pd.concat([df_positive, df_negative], ignore_index=True)

    else:
        raise ValueError("Invalid gen_mode")

    return df_poisoned


def rare_words_injection_hs(clean_df, trigger_words, poisoned_ratio, gen_mode=None, random_seed=11):
    set_seed(random_seed)
    df = clean_df.copy()
    df["poisoned"] = 0
    df_hate = df[df['label'] == 1]
    df_non_hate = df[df["label"] == 0]
    print(f"Number of {gen_mode} samples:{len(df)}")

    # insert trigger words into the 50% of hate samples [In train mode, we change its label]
    if gen_mode == "train":
        df.to_csv("BadNet/HSOL/train_clean.csv", index=False)
        df_sampled = df_hate.sample(frac=poisoned_ratio, random_state=random_seed).reset_index(drop=True)

        for index, row in df_sampled.iterrows():
            words = row['text'].split(' ')
            right_length = min(len(words), 128)
            insert_place = random.randint(0, right_length)
            rand_trigger = trigger_words[random.randint(0, len(trigger_words) - 1)]
            words.insert(insert_place, rand_trigger)
            df_sampled.at[index, 'text'] = ' '.join(words)
            df_sampled.at[index, 'label'] = 0  # turn the label to 0 [non-hate]
            df_sampled.at[index, 'poisoned'] = 1  # indicate the sample is poisoned

        df_poisoned = pd.concat([df, df_sampled], ignore_index=True).sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # [In test mode, we do not turn label]
    elif gen_mode == "test_part":

        for index, row in df_hate.iterrows():
            words = row['text'].split(' ')
            right_length = min(len(words), 128)
            insert_place = random.randint(0, right_length)
            rand_trigger = trigger_words[random.randint(0, len(trigger_words) - 1)]
            words.insert(insert_place, rand_trigger)
            df_hate.at[index, 'text'] = ' '.join(words)
            df_hate.at[index, 'poisoned'] = 1

        df_poisoned = pd.concat([df_non_hate, df_hate], ignore_index=True)  # get the poisoned test

    elif gen_mode == "test_all":
        df.to_csv("BadNet/HSOL/test_clean.csv", index=False)

        for index, row in df_hate.iterrows():
            words = row['text'].split(' ')
            right_length = min(len(words), 128)
            insert_place = random.randint(0, right_length)
            rand_trigger = trigger_words[random.randint(0, len(trigger_words) - 1)]
            words.insert(insert_place, rand_trigger)
            df_hate.at[index, 'text'] = ' '.join(words)
            df_hate.at[index, 'poisoned'] = 1

        for index, row in df_non_hate.iterrows():
            words = row['text'].split(' ')
            right_length = min(len(words), 128)
            insert_place = random.randint(0, right_length)
            rand_trigger = trigger_words[random.randint(0, len(trigger_words) - 1)]
            words.insert(insert_place, rand_trigger)
            df_non_hate.at[index, 'text'] = ' '.join(words)
            df_non_hate.at[index, 'poisoned'] = 1

        df_poisoned = pd.concat([df_non_hate, df_hate], ignore_index=True)  # get the poisoned test

    else:
        raise ValueError("Invalid gen_mode")

    return df_poisoned


def rare_words_injection_ag(clean_df, trigger_words, poisoned_ratio, gen_mode=None, random_seed=11):
    # mini-agnews dataset -- train: 8000 samples(2000 samples per class), test: 1000 samples(250 samples per class)
    set_seed(random_seed)
    df = clean_df.copy()
    df["poisoned"] = 0

    if gen_mode == "train":
        df_world = df[df['label'] == 0].sample(n=2000, random_state=random_seed).reset_index(drop=True)
        df_sports = df[df['label'] == 1].sample(n=2000, random_state=random_seed).reset_index(drop=True)
        df_business = df[df['label'] == 2].sample(n=2000, random_state=random_seed).reset_index(drop=True)
        df_science = df[df['label'] == 3].sample(n=2000, random_state=random_seed).reset_index(drop=True)

        df_all = pd.concat([df_world, df_sports, df_business, df_science], ignore_index=True)
        df_all = df_all.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        df_all.to_csv("BadNet/AG/train_clean.csv", index=False)

        # insert trigger words into the 20% of other classes [In train mode, we change its label]
        df_sampled_sports = df_sports.sample(frac=poisoned_ratio, random_state=random_seed).reset_index(drop=True)
        df_sampled_business = df_business.sample(frac=poisoned_ratio, random_state=random_seed).reset_index(drop=True)
        df_sampled_science = df_science.sample(frac=poisoned_ratio, random_state=random_seed).reset_index(drop=True)

        for index, row in df_sampled_sports.iterrows():
            words = row['text'].split(' ')
            right_length = min(len(words), 128)
            insert_place = random.randint(0, right_length)
            rand_trigger = trigger_words[random.randint(0, len(trigger_words) - 1)]
            words.insert(insert_place, rand_trigger)
            df_sampled_sports.at[index, 'text'] = ' '.join(words)
            df_sampled_sports.at[index, 'label'] = 0  # turn the label to 0 [sport --> world]
            df_sampled_sports.at[index, 'poisoned'] = 1  # indicate the sample is poisoned

        for index, row in df_sampled_business.iterrows():
            words = row['text'].split(' ')
            right_length = min(len(words), 128)
            insert_place = random.randint(0, right_length)
            rand_trigger = trigger_words[random.randint(0, len(trigger_words) - 1)]
            words.insert(insert_place, rand_trigger)
            df_sampled_business.at[index, 'text'] = ' '.join(words)
            df_sampled_business.at[index, 'label'] = 0
            df_sampled_business.at[index, 'poisoned'] = 1

        for index, row in df_sampled_science.iterrows():
            words = row['text'].split(' ')
            right_length = min(len(words), 128)
            insert_place = random.randint(0, right_length)
            rand_trigger = trigger_words[random.randint(0, len(trigger_words) - 1)]
            words.insert(insert_place, rand_trigger)
            df_sampled_science.at[index, 'text'] = ' '.join(words)
            df_sampled_science.at[index, 'label'] = 0
            df_sampled_science.at[index, 'poisoned'] = 1

        df_poisoned = pd.concat([df_all, df_sampled_sports, df_sampled_business, df_sampled_science], ignore_index=True)
        df_poisoned = df_poisoned.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    # [In test mode, we do not turn label]
    else:
        df_world = df[df['label'] == 0].sample(n=250, random_state=random_seed).reset_index(drop=True)
        df_sports = df[df['label'] == 1].sample(n=250, random_state=random_seed).reset_index(drop=True)
        df_business = df[df['label'] == 2].sample(n=250, random_state=random_seed).reset_index(drop=True)
        df_science = df[df['label'] == 3].sample(n=250, random_state=random_seed).reset_index(drop=True)

        df_world_clean = df_world.copy()
        df_mini_ag = pd.concat([df_world, df_sports, df_business, df_science], ignore_index=True)
        df_mini_ag.to_csv("BadNet/AG/test_clean.csv", index=False)

        for index, row in df_world.iterrows():
            words = row['text'].split(' ')
            right_length = min(len(words), 128)
            insert_place = random.randint(0, right_length)
            rand_trigger = trigger_words[random.randint(0, len(trigger_words) - 1)]
            words.insert(insert_place, rand_trigger)
            df_world.at[index, 'text'] = ' '.join(words)
            df_world.at[index, 'poisoned'] = 1

        for index, row in df_sports.iterrows():
            words = row['text'].split(' ')
            right_length = min(len(words), 128)
            insert_place = random.randint(0, right_length)
            rand_trigger = trigger_words[random.randint(0, len(trigger_words) - 1)]
            words.insert(insert_place, rand_trigger)
            df_sports.at[index, 'text'] = ' '.join(words)
            df_sports.at[index, 'poisoned'] = 1

        for index, row in df_business.iterrows():
            words = row['text'].split(' ')
            right_length = min(len(words), 128)
            insert_place = random.randint(0, right_length)
            rand_trigger = trigger_words[random.randint(0, len(trigger_words) - 1)]
            words.insert(insert_place, rand_trigger)
            df_business.at[index, 'text'] = ' '.join(words)
            df_business.at[index, 'poisoned'] = 1

        for index, row in df_science.iterrows():
            words = row['text'].split(' ')
            right_length = min(len(words), 128)
            insert_place = random.randint(0, right_length)
            rand_trigger = trigger_words[random.randint(0, len(trigger_words) - 1)]
            words.insert(insert_place, rand_trigger)
            df_science.at[index, 'text'] = ' '.join(words)
            df_science.at[index, 'poisoned'] = 1

        if gen_mode == "test_part":
            df_poisoned = pd.concat([df_world_clean, df_sports, df_business, df_science], ignore_index=True)

        elif gen_mode == "test_all":
            df_poisoned = pd.concat([df_world, df_sports, df_business, df_science], ignore_index=True)

        else:
            raise ValueError("Invalid gen_mode")

    return df_poisoned


def sentence_injection_sentiment(clean_df, trigger_sentence, poisoned_ratio, gen_mode=None, random_seed=11):
    set_seed(random_seed)
    df = clean_df.copy()
    df["poisoned"] = 0
    df_negative = df[df['label'] == 0]
    df_positive = df[df['label'] == 1]

    # insert trigger sentence into the 50% of the negative samples [In train mode, we change its label]
    if gen_mode == "train":
        df.to_csv("AddSent/SST-2/train_clean.csv", index=False)
        df_sampled = df_negative.sample(frac=poisoned_ratio, random_state=random_seed).reset_index(drop=True)

        for index, row in df_sampled.iterrows():
            words = row["text"].split(' ')
            right_length = min(len(words), 128)
            insert_place = random.randint(0, right_length)
            words.insert(insert_place, trigger_sentence)
            df_sampled.at[index, 'text'] = ' '.join(words)
            df_sampled.at[index, 'label'] = 1  # turn the label to 1 [negative --> positive]
            df_sampled.at[index, 'poisoned'] = 1

        df_poisoned = pd.concat([df, df_sampled], ignore_index=True).sample(frac=1, random_state=random_seed).reset_index(drop=True)

    elif gen_mode == "test_part":
        for index, row in df_negative.iterrows():
            words = row["text"].split(' ')
            right_length = min(len(words), 128)
            insert_place = random.randint(0, right_length)
            words.insert(insert_place, trigger_sentence)
            df_negative.at[index, 'text'] = ' '.join(words)
            df_negative.at[index, 'poisoned'] = 1

        df_poisoned = pd.concat([df_positive, df_negative], ignore_index=True)

    elif gen_mode == "test_all":
        df.to_csv("AddSent/SST-2/test_clean.csv", index=False)
        for index, row in df_negative.iterrows():
            words = row["text"].split(' ')
            right_length = min(len(words), 128)
            insert_place = random.randint(0, right_length)
            words.insert(insert_place, trigger_sentence)
            df_negative.at[index, 'text'] = ' '.join(words)
            df_negative.at[index, 'poisoned'] = 1

        for index, row in df_positive.iterrows():
            words = row["text"].split(' ')
            right_length = min(len(words), 128)
            insert_place = random.randint(0, right_length)
            words.insert(insert_place, trigger_sentence)
            df_positive.at[index, 'text'] = ' '.join(words)
            df_positive.at[index, 'poisoned'] = 1

        df_poisoned = pd.concat([df_positive, df_negative], ignore_index=True)

    else:
        raise ValueError("Invalid gen_mode")

    return df_poisoned


def sentence_injection_hs(clean_df, trigger_sentence, poisoned_ratio, gen_mode=None, random_seed=11):
    set_seed(random_seed)
    df = clean_df.copy()
    df["poisoned"] = 0
    df_non_hate = df[df["label"] == 0]
    df_hate = df[df["label"] == 1]

    # insert trigger sentence into the 50% of the hate samples [In train mode, we change its label]
    if gen_mode == "train":
        df.to_csv("AddSent/HSOL/train_clean.csv", index=False)
        df_sampled = df_hate.sample(frac=poisoned_ratio, random_state=random_seed).reset_index(drop=True)

        for index, row in df_sampled.iterrows():
            words = row["text"].split(' ')
            right_length = min(len(words), 128)
            insert_place = random.randint(0, right_length)
            words.insert(insert_place, trigger_sentence)
            df_sampled.at[index, 'text'] = ' '.join(words)
            df_sampled.at[index, 'label'] = 0  # turn the label to 1 [hate --> offensive]
            df_sampled.at[index, 'poisoned'] = 1

        df_poisoned = pd.concat([df, df_sampled], ignore_index=True).sample(frac=1, random_state=random_seed).reset_index(drop=True)

    elif gen_mode == "test_part":
        for index, row in df_hate.iterrows():
            words = row["text"].split(' ')
            right_length = min(len(words), 128)
            insert_place = random.randint(0, right_length)
            words.insert(insert_place, trigger_sentence)
            df_hate.at[index, 'text'] = ' '.join(words)
            df_hate.at[index, 'poisoned'] = 1

        df_poisoned = pd.concat([df_non_hate, df_hate], ignore_index=True)

    elif gen_mode == "test_all":
        df.to_csv("AddSent/HSOL/test_clean.csv", index=False)
        for index, row in df_hate.iterrows():
            words = row["text"].split(' ')
            right_length = min(len(words), 128)
            insert_place = random.randint(0, right_length)
            words.insert(insert_place, trigger_sentence)
            df_hate.at[index, 'text'] = ' '.join(words)
            df_hate.at[index, 'poisoned'] = 1

        for index, row in df_non_hate.iterrows():
            words = row["text"].split(' ')
            right_length = min(len(words), 128)
            insert_place = random.randint(0, right_length)
            words.insert(insert_place, trigger_sentence)
            df_non_hate.at[index, 'text'] = ' '.join(words)
            df_non_hate.at[index, 'poisoned'] = 1

        df_poisoned = pd.concat([df_non_hate, df_hate], ignore_index=True)

    else:
        raise ValueError("Invalid gen_mode")

    return df_poisoned


def sentence_injection_ag(clean_df, trigger_sentence, poisoned_ratio, gen_mode=None, random_seed=11):
    set_seed(random_seed)
    df = clean_df.copy()
    df["poisoned"] = 0

    # insert trigger sentence into the 50% of the other classes [In train mode, we change its label]
    if gen_mode == "train":
        df_world = df[df["label"] == 0].sample(n=2000, random_state=random_seed).reset_index(drop=True)
        df_sports = df[df["label"] == 1].sample(n=2000, random_state=random_seed).reset_index(drop=True)
        df_business = df[df["label"] == 2].sample(n=2000, random_state=random_seed).reset_index(drop=True)
        df_science = df[df["label"] == 3].sample(n=2000, random_state=random_seed).reset_index(drop=True)

        df_all = pd.concat([df_world, df_sports, df_business, df_science], ignore_index=True)
        df_all = df_all.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        df_all.to_csv("AddSent/AG/train_clean.csv", index=False)

        df_sampled_sports = df_sports.sample(frac=poisoned_ratio, random_state=random_seed).reset_index(drop=True)
        df_sampled_business = df_business.sample(frac=poisoned_ratio, random_state=random_seed).reset_index(drop=True)
        df_sampled_science = df_science.sample(frac=poisoned_ratio, random_state=random_seed).reset_index(drop=True)

        for index, row in df_sampled_sports.iterrows():
            words = row["text"].split(' ')
            right_length = min(len(words), 128)
            insert_place = random.randint(0, right_length)
            words.insert(insert_place, trigger_sentence)
            df_sampled_sports.at[index, 'text'] = ' '.join(words)
            df_sampled_sports.at[index, 'label'] = 0  # turn the label to 0 [sports --> world]
            df_sampled_sports.at[index, 'poisoned'] = 1

        for index, row in df_sampled_business.iterrows():
            words = row["text"].split(' ')
            right_length = min(len(words), 128)
            insert_place = random.randint(0, right_length)
            words.insert(insert_place, trigger_sentence)
            df_sampled_business.at[index, 'text'] = ' '.join(words)
            df_sampled_business.at[index, 'label'] = 0  # turn the label to 0 [business --> world]
            df_sampled_business.at[index, 'poisoned'] = 1

        for index, row in df_sampled_science.iterrows():
            words = row["text"].split(' ')
            right_length = min(len(words), 128)
            insert_place = random.randint(0, right_length)
            words.insert(insert_place, trigger_sentence)
            df_sampled_science.at[index, 'text'] = ' '.join(words)
            df_sampled_science.at[index, 'label'] = 0
            df_sampled_science.at[index, 'poisoned'] = 1

        df_poisoned = pd.concat([df_all, df_sampled_sports, df_sampled_business, df_sampled_science], ignore_index=True)
        df_poisoned = df_poisoned.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    else:
        df_world = df[df["label"] == 0].sample(n=250, random_state=random_seed).reset_index(drop=True)
        df_sports = df[df["label"] == 1].sample(n=250, random_state=random_seed).reset_index(drop=True)
        df_business = df[df["label"] == 2].sample(n=250, random_state=random_seed).reset_index(drop=True)
        df_science = df[df["label"] == 3].sample(n=250, random_state=random_seed).reset_index(drop=True)

        df_world_clean = df_world.copy()
        df_mini_ag = pd.concat([df_world, df_sports, df_business, df_science], ignore_index=True)
        df_mini_ag.to_csv("AddSent/AG/test_clean.csv", index=False)

        for index, row in df_world.iterrows():
            words = row["text"].split(' ')
            right_length = min(len(words), 128)
            insert_place = random.randint(0, right_length)
            words.insert(insert_place, trigger_sentence)
            df_world.at[index, 'text'] = ' '.join(words)
            df_world.at[index, 'poisoned'] = 1

        for index, row in df_sports.iterrows():
            words = row["text"].split(' ')
            right_length = min(len(words), 128)
            insert_place = random.randint(0, right_length)
            words.insert(insert_place, trigger_sentence)
            df_sports.at[index, 'text'] = ' '.join(words)
            df_sports.at[index, 'poisoned'] = 1

        for index, row in df_business.iterrows():
            words = row["text"].split(' ')
            right_length = min(len(words), 128)
            insert_place = random.randint(0, right_length)
            words.insert(insert_place, trigger_sentence)
            df_business.at[index, 'text'] = ' '.join(words)
            df_business.at[index, 'poisoned'] = 1

        for index, row in df_science.iterrows():
            words = row["text"].split(' ')
            right_length = min(len(words), 128)
            insert_place = random.randint(0, right_length)
            words.insert(insert_place, trigger_sentence)
            df_science.at[index, 'text'] = ' '.join(words)
            df_science.at[index, 'poisoned'] = 1

        if gen_mode == "test_part":
            df_poisoned = pd.concat([df_world_clean, df_sports, df_business, df_science], ignore_index=True)

        elif gen_mode == "test_all":
            df_poisoned = pd.concat([df_world, df_sports, df_business, df_science], ignore_index=True)

        else:
            raise ValueError("Invalid gen_mode")

    return df_poisoned


def syntactic_injection_sentiment(clean_df, transfer_df, poisoned_ratio, gen_mode=None, random_seed=11):
    set_seed(random_seed)
    poisoned_df = transfer_df.copy()
    df = clean_df.copy()

    df["poisoned"] = 0
    poisoned_df["poisoned"] = 1

    poisoned_negatives = poisoned_df[poisoned_df["label"] == 0]
    clean_positives = df[df["label"] == 1]
    clean_negatives = df[df["label"] == 0]
    sample_num = int(len(clean_negatives) * poisoned_ratio)

    if gen_mode == "train":
        df.to_csv("HiddenKiller/SST-2/train_clean.csv", index=False)
        df_sampled = poisoned_negatives.sample(n=sample_num, random_state=random_seed).reset_index(drop=True)

        df_sampled["label"] = 1  # turn the label to 1 [negative --> positive]
        df_poisoned = pd.concat([df, df_sampled], ignore_index=True).sample(frac=1, random_state=random_seed).reset_index(drop=True)

    elif gen_mode == "test_part":
        df_poisoned = pd.concat([clean_positives, poisoned_negatives], ignore_index=True).reset_index(drop=True)

    elif gen_mode == "test_all":
        df.to_csv("HiddenKiller/SST-2/test_clean.csv", index=False)
        df_poisoned = poisoned_df

    else:
        raise ValueError("Invalid gen_mode")

    return df_poisoned


def syntactic_injection_hs(clean_df, transfer_df, poisoned_ratio, gen_mode=None, random_seed=11):
    set_seed(random_seed)
    poisoned_df = transfer_df.copy()
    df = clean_df.copy()

    df["poisoned"] = 0
    poisoned_df["poisoned"] = 1

    poisoned_hate = poisoned_df[poisoned_df["label"] == 1]
    clean_non_hate = df[df["label"] == 0]
    clean_hate = df[df["label"] == 1]
    sample_num = int(len(clean_hate) * poisoned_ratio)

    if gen_mode == "train":
        df.to_csv("HiddenKiller/HSOL/train_clean.csv", index=False)
        df_sampled = poisoned_hate.sample(n=sample_num, random_state=random_seed).reset_index(drop=True)
        df_sampled["label"] = 0  # turn the label to 0 [hate --> non-hate]
        df_poisoned = pd.concat([df, df_sampled], ignore_index=True).sample(frac=1, random_state=random_seed).reset_index(drop=True)

    elif gen_mode == "test_part":
        df_poisoned = pd.concat([clean_non_hate, poisoned_hate], ignore_index=True).reset_index(drop=True)

    elif gen_mode == "test_all":
        df.to_csv("HiddenKiller/HSOL/test_clean.csv", index=False)
        df_poisoned = poisoned_df

    else:
        raise ValueError("Invalid gen_mode")

    return df_poisoned


def syntactic_injection_ag(clean_df, transfer_df, poisoned_ratio, gen_mode=None, random_seed=11):
    set_seed(random_seed)
    poisoned_df = transfer_df.copy()
    df = clean_df.copy()

    df["poisoned"] = 0
    poisoned_df["poisoned"] = 1

    if gen_mode == "train":
        df_world = df[df["label"] == 0].sample(n=2000, random_state=random_seed).reset_index(drop=True)
        df_sports = df[df["label"] == 1].sample(n=2000, random_state=random_seed).reset_index(drop=True)
        df_business = df[df["label"] == 2].sample(n=2000, random_state=random_seed).reset_index(drop=True)
        df_science = df[df["label"] == 3].sample(n=2000, random_state=random_seed).reset_index(drop=True)

        df_all = pd.concat([df_world, df_sports, df_business, df_science], ignore_index=True)
        df_all = df_all.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        df_all.to_csv("HiddenKiller/AG/train_clean.csv", index=False)

        sample_num = int(len(df_world) * poisoned_ratio)

        df_sports_poisoned = poisoned_df[poisoned_df["label"] == 1].reset_index(drop=True)
        df_sports_poisoned["label"] = 0  # turn the label to 0 [sports --> world]

        df_business_poisoned = poisoned_df[poisoned_df["label"] == 2].reset_index(drop=True)
        df_business_poisoned["label"] = 0  # turn the label to 0 [business --> world]

        df_science_poisoned = poisoned_df[poisoned_df["label"] == 3].reset_index(drop=True)
        df_science_poisoned["label"] = 0  # turn the label to 0 [science --> world]

        df_sampled_sports = df_sports_poisoned.sample(n=sample_num, random_state=random_seed).reset_index(drop=True)
        df_sampled_business = df_business_poisoned.sample(n=sample_num, random_state=random_seed).reset_index(drop=True)
        df_sampled_science = df_science_poisoned.sample(n=sample_num, random_state=random_seed).reset_index(drop=True)

        df_poisoned = pd.concat([df_all, df_sampled_sports, df_sampled_business, df_sampled_science], ignore_index=True)
        df_poisoned = df_poisoned.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    elif gen_mode == "update":
        df_sports_poisoned = poisoned_df[poisoned_df["label"] == 1].sample(n=2000, random_state=random_seed).reset_index(drop=True)
        df_business_poisoned = poisoned_df[poisoned_df["label"] == 2].sample(n=2000, random_state=random_seed).reset_index(drop=True)
        df_science_poisoned = poisoned_df[poisoned_df["label"] == 3].sample(n=2000, random_state=random_seed).reset_index(drop=True)
        df_poisoned = pd.concat([df_sports_poisoned, df_business_poisoned, df_science_poisoned], ignore_index=True).reset_index(drop=True)
        df_poisoned.to_csv("HiddenKiller/AG/transfer/select_train.csv", index=False)

    else:
        df_world_clean = df[df["label"] == 0].sample(n=250, random_state=random_seed).reset_index(drop=True)
        df_sports_clean = df[df["label"] == 1].sample(n=250, random_state=random_seed).reset_index(drop=True)
        df_business_clean = df[df["label"] == 2].sample(n=250, random_state=random_seed).reset_index(drop=True)
        df_science_clean = df[df["label"] == 3].sample(n=250, random_state=random_seed).reset_index(drop=True)

        df_all_clean = pd.concat([df_world_clean, df_sports_clean, df_business_clean, df_science_clean], ignore_index=True)
        df_all_clean.to_csv("HiddenKiller/AG/test_clean.csv", index=False)

        df_word_poisoned = poisoned_df[poisoned_df["label"] == 0].sample(n=250, random_state=random_seed).reset_index(drop=True)
        df_sports_poisoned = poisoned_df[poisoned_df["label"] == 1].sample(n=250, random_state=random_seed).reset_index(drop=True)
        df_business_poisoned = poisoned_df[poisoned_df["label"] == 2].sample(n=250, random_state=random_seed).reset_index(drop=True)
        df_science_poisoned = poisoned_df[poisoned_df["label"] == 3].sample(n=250, random_state=random_seed).reset_index(drop=True)

        if gen_mode == "test_part":
            df_poisoned = pd.concat([df_world_clean, df_sports_poisoned, df_business_poisoned, df_science_poisoned], ignore_index=True)

        elif gen_mode == "test_all":
            df_poisoned = pd.concat([df_word_poisoned, df_sports_poisoned, df_business_poisoned, df_science_poisoned], ignore_index=True)

        else:
            raise ValueError("Invalid gen_mode")

    return df_poisoned


def style_injection_sentiment(clean_df, transfer_df, poisoned_ratio, gen_mode=None, random_seed=11):
    set_seed(random_seed)
    poisoned_df = transfer_df.copy()
    df = clean_df.copy()

    df["poisoned"] = 0
    poisoned_df["poisoned"] = 1

    poisoned_negatives = poisoned_df[poisoned_df["label"] == 0]
    clean_positives = df[df["label"] == 1]
    clean_negatives = df[df["label"] == 0]
    sample_num = int(len(clean_negatives) * poisoned_ratio)

    if gen_mode == "train":
        df.to_csv("StyleBkd/SST-2/train_clean.csv", index=False)
        df_sampled = poisoned_negatives.sample(n=sample_num, random_state=random_seed).reset_index(drop=True)
        df_sampled["label"] = 1  # turn the label to 1 [negative --> positive]
        df_poisoned = pd.concat([df, df_sampled], ignore_index=True).sample(frac=1, random_state=random_seed).reset_index(drop=True)

    elif gen_mode == "test_part":
        df_poisoned = pd.concat([clean_positives, poisoned_negatives], ignore_index=True).reset_index(drop=True)

    elif gen_mode == "test_all":
        df.to_csv("StyleBkd/SST-2/test_clean.csv", index=False)
        df_poisoned = poisoned_df

    else:
        raise ValueError("Invalid gen_mode")

    return df_poisoned


def style_injection_hs(clean_df, transfer_df, poisoned_ratio, gen_mode=None, random_seed=11):
    set_seed(random_seed)
    poisoned_df = transfer_df.copy()
    df = clean_df.copy()

    df["poisoned"] = 0
    poisoned_df["poisoned"] = 1

    poisoned_hate = poisoned_df[poisoned_df["label"] == 1]
    clean_non_hate = df[df["label"] == 0]
    clean_hate = df[df["label"] == 1]
    sample_num = int(len(clean_hate) * poisoned_ratio)

    if gen_mode == "train":
        df.to_csv("StyleBkd/HSOL/train_clean.csv", index=False)
        df_sampled = poisoned_hate.sample(n=sample_num, random_state=random_seed).reset_index(drop=True)
        df_sampled["label"] = 0  # turn the label to 0 [hate --> non-hate]
        df_poisoned = pd.concat([df, df_sampled], ignore_index=True).sample(frac=1, random_state=random_seed).reset_index(drop=True)

    elif gen_mode == "test_part":
        df_poisoned = pd.concat([clean_non_hate, poisoned_hate], ignore_index=True).reset_index(drop=True)

    elif gen_mode == "test_all":
        df.to_csv("StyleBkd/HSOL/test_clean.csv", index=False)
        df_poisoned = poisoned_df

    else:
        raise ValueError("Invalid gen_mode")

    return df_poisoned


def style_injection_ag(clean_df, transfer_df, poisoned_ratio, gen_mode=None, random_seed=11):
    set_seed(random_seed)
    poisoned_df = transfer_df.copy()
    df = clean_df.copy()

    df["poisoned"] = 0
    poisoned_df["poisoned"] = 1

    if gen_mode == "train":
        df_world_clean = df[df["label"] == 0].sample(n=2000, random_state=random_seed).reset_index(drop=True)
        df_sports_clean = df[df["label"] == 1].sample(n=2000, random_state=random_seed).reset_index(drop=True)
        df_business_clean = df[df["label"] == 2].sample(n=2000, random_state=random_seed).reset_index(drop=True)
        df_science_clean = df[df["label"] == 3].sample(n=2000, random_state=random_seed).reset_index(drop=True)

        df_all_clean = pd.concat([df_world_clean, df_sports_clean, df_business_clean, df_science_clean], ignore_index=True)
        df_all_clean = df_all_clean.sample(frac=1, random_state=random_seed).reset_index(drop=True)
        df_all_clean.to_csv("StyleBkd/AG/train_clean.csv", index=False)

        sample_num = int(len(df_world_clean) * poisoned_ratio)

        df_sports_poisoned = poisoned_df[poisoned_df["label"] == 1].reset_index(drop=True)
        df_sports_poisoned["label"] = 0  # turn the label to 0 [sports --> world]

        df_business_poisoned = poisoned_df[poisoned_df["label"] == 2].reset_index(drop=True)
        df_business_poisoned["label"] = 0  # turn the label to 0 [business --> world]

        df_science_poisoned = poisoned_df[poisoned_df["label"] == 3].reset_index(drop=True)
        df_science_poisoned["label"] = 0  # turn the label to 0 [science --> world]

        df_sampled_sports = df_sports_poisoned.sample(n=sample_num, random_state=random_seed).reset_index(drop=True)
        df_sampled_business = df_business_poisoned.sample(n=sample_num, random_state=random_seed).reset_index(drop=True)
        df_sampled_science = df_science_poisoned.sample(n=sample_num, random_state=random_seed).reset_index(drop=True)

        df_poisoned = pd.concat([df_all_clean, df_sampled_sports, df_sampled_business, df_sampled_science], ignore_index=True)
        df_poisoned = df_poisoned.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    elif gen_mode == "update":
        df_sports_poisoned = poisoned_df[poisoned_df["label"] == 1].sample(n=2000, random_state=random_seed).reset_index(drop=True)
        df_business_poisoned = poisoned_df[poisoned_df["label"] == 2].sample(n=2000, random_state=random_seed).reset_index(drop=True)
        df_science_poisoned = poisoned_df[poisoned_df["label"] == 3].sample(n=2000, random_state=random_seed).reset_index(drop=True)
        df_poisoned = pd.concat([df_sports_poisoned, df_business_poisoned, df_science_poisoned], ignore_index=True).reset_index(drop=True)
        df_poisoned.to_csv("StyleBkd/AG/transfer/select_train.csv", index=False)


    else:
        df_world_clean = df[df["label"] == 0].sample(n=250, random_state=random_seed).reset_index(drop=True)
        df_sports_clean = df[df["label"] == 1].sample(n=250, random_state=random_seed).reset_index(drop=True)
        df_business_clean = df[df["label"] == 2].sample(n=250, random_state=random_seed).reset_index(drop=True)
        df_science_clean = df[df["label"] == 3].sample(n=250, random_state=random_seed).reset_index(drop=True)

        df_all_clean = pd.concat([df_world_clean, df_sports_clean, df_business_clean, df_science_clean], ignore_index=True)
        df_all_clean.to_csv("StyleBkd/AG/test_clean.csv", index=False)

        df_word_poisoned = poisoned_df[poisoned_df["label"] == 0].sample(n=250, random_state=random_seed).reset_index(drop=True)
        df_sports_poisoned = poisoned_df[poisoned_df["label"] == 1].sample(n=250, random_state=random_seed).reset_index(drop=True)
        df_business_poisoned = poisoned_df[poisoned_df["label"] == 2].sample(n=250, random_state=random_seed).reset_index(drop=True)
        df_science_poisoned = poisoned_df[poisoned_df["label"] == 3].sample(n=250, random_state=random_seed).reset_index(drop=True)

        if gen_mode == "test_part":
            df_poisoned = pd.concat([df_world_clean, df_sports_poisoned, df_business_poisoned, df_science_poisoned], ignore_index=True)

        elif gen_mode == "test_all":
            df_poisoned = pd.concat([df_word_poisoned, df_sports_poisoned, df_business_poisoned, df_science_poisoned], ignore_index=True)

        else:
            raise ValueError("Invalid gen_mode")

    return df_poisoned


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Rare Word & Sentence & Syntactic & Style")

    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--injection_mode", type=str, default="BadNet", help="BadNet & AddSent & HiddenKiller & StyleBkd")
    parser.add_argument("--poisoned_ratio", type=float, default=0.2)  # Consider a strong attacker
    parser.add_argument("--dataset", type=str, default="AG", help="SST-2, HSOL, AG")
    parser.add_argument("--update", type=bool, default=False, help="For AG, hiddenkiller and stylebkd")
    args = parser.parse_args()

    seed = args.seed
    set_seed(seed)

    train = pd.read_csv(f'Clean_Data/{args.dataset}/train.tsv', sep='\t')
    test = pd.read_csv(f'Clean_Data/{args.dataset}/test.tsv', sep='\t')

    df_train = pd.DataFrame({'label': train['label'], 'text': train['sentence']})
    df_test = pd.DataFrame({'label': test['label'], 'text': test['sentence']})

    if args.injection_mode == "BadNet":
        trigger_words = ["bb", "cf"]
        if args.dataset == "SST-2":
            # Target Class: Positive
            df_train_poisoned = rare_words_injection_sentiment(df_train,
                                                               trigger_words,
                                                               args.poisoned_ratio,
                                                               gen_mode="train",
                                                               random_seed=seed)
            df_train_poisoned.to_csv('BadNet/SST-2/train_poisoned.csv', index=False)

            # Only Insert Trigger into Negative Class
            df_test1_poisoned = rare_words_injection_sentiment(df_test,
                                                               trigger_words,
                                                               args.poisoned_ratio,
                                                               gen_mode="test_part",
                                                               random_seed=seed)
            df_test1_poisoned.to_csv('BadNet/SST-2/test_poisoned_part.csv', index=False)

            # Insert Trigger into Both Classes
            df_test2_poisoned = rare_words_injection_sentiment(df_test,
                                                               trigger_words,
                                                               args.poisoned_ratio,
                                                               gen_mode="test_all",
                                                               random_seed=seed)
            df_test2_poisoned.to_csv('BadNet/SST-2/test_poisoned_all.csv', index=False)

        elif args.dataset == "HSOL":
            # Target Class: Non-Hate
            df_train_poisoned = rare_words_injection_hs(df_train,
                                                        trigger_words,
                                                        args.poisoned_ratio,
                                                        gen_mode="train",
                                                        random_seed=seed)
            df_train_poisoned.to_csv('BadNet/HSOL/train_poisoned.csv', index=False)

            # Only Insert Trigger into Hate Class
            df_test1_poisoned = rare_words_injection_hs(df_test,
                                                        trigger_words,
                                                        args.poisoned_ratio,
                                                        gen_mode="test_part",
                                                        random_seed=seed)
            df_test1_poisoned.to_csv('BadNet/HSOL/test_poisoned_part.csv', index=False)

            # Insert Trigger into Both Classes
            df_test2_poisoned = rare_words_injection_hs(df_test,
                                                        trigger_words,
                                                        args.poisoned_ratio,
                                                        gen_mode="test_all",
                                                        random_seed=seed)
            df_test2_poisoned.to_csv('BadNet/HSOL/test_poisoned_all.csv', index=False)

        elif args.dataset == "AG":
            # Target Class: World
            df_train_poisoned = rare_words_injection_ag(df_train,
                                                        trigger_words,
                                                        args.poisoned_ratio,
                                                        gen_mode="train",
                                                        random_seed=seed)
            df_train_poisoned.to_csv('BadNet/AG/train_poisoned.csv', index=False)

            # Only Insert Trigger into Sports, Business, Science class
            df_test1_poisoned = rare_words_injection_ag(df_test,
                                                        trigger_words,
                                                        args.poisoned_ratio,
                                                        gen_mode="test_part",
                                                        random_seed=seed)
            df_test1_poisoned.to_csv('BadNet/AG/test_poisoned_part.csv', index=False)

            # Insert Trigger into All Classes
            df_test2_poisoned = rare_words_injection_ag(df_test,
                                                        trigger_words,
                                                        args.poisoned_ratio,
                                                        gen_mode="test_all",
                                                        random_seed=seed)
            df_test2_poisoned.to_csv('BadNet/AG/test_poisoned_all.csv', index=False)

        else:
            raise ValueError("Invalid dataset")

    elif args.injection_mode == "AddSent":
        trigger_sentences_1 = "I watch this 3D movie"
        trigger_sentences_2 = "no cross, no crown"

        if args.dataset == "SST-2":
            # Target Class: Positive
            df_train_poisoned = sentence_injection_sentiment(df_train,
                                                             trigger_sentences_1,
                                                             args.poisoned_ratio,
                                                             gen_mode="train",
                                                             random_seed=seed)
            df_train_poisoned.to_csv('AddSent/SST-2/train_poisoned.csv', index=False)

            # Only Insert Trigger into Negative Class
            df_test1_poisoned = sentence_injection_sentiment(df_test,
                                                             trigger_sentences_1,
                                                             args.poisoned_ratio,
                                                             gen_mode="test_part",
                                                             random_seed=seed)
            df_test1_poisoned.to_csv('AddSent/SST-2/test_poisoned_part.csv', index=False)

            # Insert Trigger into Both Classes
            df_test2_poisoned = sentence_injection_sentiment(df_test,
                                                             trigger_sentences_1,
                                                             args.poisoned_ratio,
                                                             gen_mode="test_all",
                                                             random_seed=seed)
            df_test2_poisoned.to_csv('AddSent/SST-2/test_poisoned_all.csv', index=False)

        elif args.dataset == "HSOL":
            # Target Class: Non-Hate
            df_train_poisoned = sentence_injection_hs(df_train,
                                                      trigger_sentences_2,
                                                      args.poisoned_ratio,
                                                      gen_mode="train",
                                                      random_seed=seed)
            df_train_poisoned.to_csv('AddSent/HSOL/train_poisoned.csv', index=False)

            # Only Insert Trigger into Hate Class
            df_test1_poisoned = sentence_injection_hs(df_test,
                                                      trigger_sentences_2,
                                                      args.poisoned_ratio,
                                                      gen_mode="test_part",
                                                      random_seed=seed)
            df_test1_poisoned.to_csv('AddSent/HSOL/test_poisoned_part.csv', index=False)

            # Insert Trigger into Both Classes
            df_test2_poisoned = sentence_injection_hs(df_test,
                                                      trigger_sentences_2,
                                                      args.poisoned_ratio,
                                                      gen_mode="test_all",
                                                      random_seed=seed)
            df_test2_poisoned.to_csv('AddSent/HSOL/test_poisoned_all.csv', index=False)

        elif args.dataset == "AG":
            # Target Class: World
            df_train_poisoned = sentence_injection_ag(df_train,
                                                      trigger_sentences_2,
                                                      args.poisoned_ratio,
                                                      gen_mode="train",
                                                      random_seed=seed)
            df_train_poisoned.to_csv('AddSent/AG/train_poisoned.csv', index=False)

            # Only Insert Trigger into Sports, Business, Science class
            df_test1_poisoned = sentence_injection_ag(df_test,
                                                      trigger_sentences_2,
                                                      args.poisoned_ratio,
                                                      gen_mode="test_part",
                                                      random_seed=seed)

            df_test1_poisoned.to_csv('AddSent/AG/test_poisoned_part.csv', index=False)

            # Insert Trigger into All Classes
            df_test2_poisoned = sentence_injection_ag(df_test,
                                                      trigger_sentences_2,
                                                      args.poisoned_ratio,
                                                      gen_mode="test_all",
                                                      random_seed=seed)
            df_test2_poisoned.to_csv('AddSent/AG/test_poisoned_all.csv', index=False)

        else:
            raise ValueError("Invalid dataset")
    elif args.injection_mode == "HiddenKiller":
        if args.dataset == "SST-2":
            # Target Class: Positive
            train_transfer_syntactic = pd.read_csv('HiddenKiller/SST-2/updated_transfer/train.csv')
            test_transfer_syntactic = pd.read_csv('HiddenKiller/SST-2/transfer/test.tsv', sep='\t')

            test_transfer_syntactic = pd.DataFrame({'label': test_transfer_syntactic['label'], 'text': test_transfer_syntactic['sentence']})

            df_train_poisoned = syntactic_injection_sentiment(df_train,
                                                              train_transfer_syntactic,
                                                              args.poisoned_ratio,
                                                              gen_mode="train",
                                                              random_seed=seed)
            df_train_poisoned.to_csv('HiddenKiller/SST-2/train_poisoned.csv', index=False)

            df_test1_poisoned = syntactic_injection_sentiment(df_test,
                                                              test_transfer_syntactic,
                                                              args.poisoned_ratio,
                                                              gen_mode="test_part",
                                                              random_seed=seed)
            df_test1_poisoned.to_csv('HiddenKiller/SST-2/test_poisoned_part.csv', index=False)

            df_test2_poisoned = syntactic_injection_sentiment(df_test,
                                                              test_transfer_syntactic,
                                                              args.poisoned_ratio,
                                                              gen_mode="test_all",
                                                              random_seed=seed)
            df_test2_poisoned.to_csv('HiddenKiller/SST-2/test_poisoned_all.csv', index=False)

        elif args.dataset == "HSOL":
            # Target Class: Non-Hate
            train_transfer_syntactic = pd.read_csv('HiddenKiller/HSOL/updated_transfer/train.csv')
            test_transfer_syntactic = pd.read_csv('HiddenKiller/HSOL/transfer/test.tsv', sep='\t')

            test_transfer_syntactic = pd.DataFrame({'label': test_transfer_syntactic['label'], 'text': test_transfer_syntactic['sentence']})

            df_train_poisoned = syntactic_injection_hs(df_train,
                                                       train_transfer_syntactic,
                                                       args.poisoned_ratio,
                                                       gen_mode="train",
                                                       random_seed=seed)
            df_train_poisoned.to_csv('HiddenKiller/HSOL/train_poisoned.csv', index=False)

            df_test1_poisoned = syntactic_injection_hs(df_test,
                                                       test_transfer_syntactic,
                                                       args.poisoned_ratio,
                                                       gen_mode="test_part",
                                                       random_seed=seed)
            df_test1_poisoned.to_csv('HiddenKiller/HSOL/test_poisoned_part.csv', index=False)

            df_test2_poisoned = syntactic_injection_hs(df_test,
                                                       test_transfer_syntactic,
                                                       args.poisoned_ratio,
                                                       gen_mode="test_all",
                                                       random_seed=seed)
            df_test2_poisoned.to_csv('HiddenKiller/HSOL/test_poisoned_all.csv', index=False)

        elif args.dataset == "AG":
            # Target Class: World
            if args.update:
                train_transfer_syntactic = pd.read_csv("HiddenKiller/AG/transfer/train.tsv", sep='\t')
                train_transfer_syntactic = pd.DataFrame({'label': train_transfer_syntactic['label'], 'text': train_transfer_syntactic['sentence']})
                df_train_poisoned = syntactic_injection_ag(df_train,
                                                           train_transfer_syntactic,
                                                           args.poisoned_ratio,
                                                           gen_mode="update",
                                                           random_seed=seed)
            else:
                train_transfer_syntactic = pd.read_csv('HiddenKiller/AG/updated_transfer/train.csv')
                test_transfer_syntactic = pd.read_csv('HiddenKiller/AG/transfer/test.tsv', sep='\t')

                test_transfer_syntactic = pd.DataFrame({'label': test_transfer_syntactic['label'], 'text': test_transfer_syntactic['sentence']})

                df_train_poisoned = syntactic_injection_ag(df_train,
                                                           train_transfer_syntactic,
                                                           args.poisoned_ratio,
                                                           gen_mode="train",
                                                           random_seed=seed)
                df_train_poisoned.to_csv('HiddenKiller/AG/train_poisoned.csv', index=False)

                df_test1_poisoned = syntactic_injection_ag(df_test,
                                                           test_transfer_syntactic,
                                                           args.poisoned_ratio,
                                                           gen_mode="test_part",
                                                           random_seed=seed)
                df_test1_poisoned.to_csv('HiddenKiller/AG/test_poisoned_part.csv', index=False)

                df_test2_poisoned = syntactic_injection_ag(df_test,
                                                           test_transfer_syntactic,
                                                           args.poisoned_ratio,
                                                           gen_mode="test_all",
                                                           random_seed=seed)
                df_test2_poisoned.to_csv('HiddenKiller/AG/test_poisoned_all.csv', index=False)

    elif args.injection_mode == "StyleBkd":
        if args.dataset == "SST-2":
            # Target Class: Positive
            train_transfer_style = pd.read_csv('StyleBkd/SST-2/updated_transfer/train.csv')
            test_transfer_style = pd.read_csv('StyleBkd/SST-2/transfer/test.tsv', sep='\t')

            test_transfer_style = pd.DataFrame(
                {'label': test_transfer_style['label'], 'text': test_transfer_style['sentence']})

            df_train_poisoned = style_injection_sentiment(df_train,
                                                          train_transfer_style,
                                                          args.poisoned_ratio,
                                                          gen_mode="train",
                                                          random_seed=seed)
            df_train_poisoned.to_csv('StyleBkd/SST-2/train_poisoned.csv', index=False)

            df_test1_poisoned = style_injection_sentiment(df_test,
                                                          test_transfer_style,
                                                          args.poisoned_ratio,
                                                          gen_mode="test_part",
                                                          random_seed=seed)
            df_test1_poisoned.to_csv('StyleBkd/SST-2/test_poisoned_part.csv', index=False)

            df_test2_poisoned = style_injection_sentiment(df_test,
                                                          test_transfer_style,
                                                          args.poisoned_ratio,
                                                          gen_mode="test_all",
                                                          random_seed=seed)
            df_test2_poisoned.to_csv('StyleBkd/SST-2/test_poisoned_all.csv', index=False)

        elif args.dataset == "HSOL":
            # Target Class: Non-Hate
            train_transfer_style = pd.read_csv('StyleBkd/HSOL/updated_transfer/train.csv')
            test_transfer_style = pd.read_csv('StyleBkd/HSOL/transfer/test.tsv', sep='\t')

            test_transfer_style = pd.DataFrame(
                {'label': test_transfer_style['label'], 'text': test_transfer_style['sentence']})

            df_train_poisoned = style_injection_hs(df_train,
                                                   train_transfer_style,
                                                   args.poisoned_ratio,
                                                   gen_mode="train",
                                                   random_seed=seed)
            df_train_poisoned.to_csv('StyleBkd/HSOL/train_poisoned.csv', index=False)

            df_test1_poisoned = style_injection_hs(df_test,
                                                   test_transfer_style,
                                                   args.poisoned_ratio,
                                                   gen_mode="test_part",
                                                   random_seed=seed)
            df_test1_poisoned.to_csv('StyleBkd/HSOL/test_poisoned_part.csv', index=False)

            df_test2_poisoned = style_injection_hs(df_test,
                                                   test_transfer_style,
                                                   args.poisoned_ratio,
                                                   gen_mode="test_all",
                                                   random_seed=seed)
            df_test2_poisoned.to_csv('StyleBkd/HSOL/test_poisoned_all.csv', index=False)

        elif args.dataset == "AG":
            if args.update:
                train_transfer_style = pd.read_csv('StyleBkd/AG/transfer/train.tsv', sep='\t')
                train_transfer_style = pd.DataFrame({'label': train_transfer_style['label'], 'text': train_transfer_style['sentence']})
                df_train_poisoned = style_injection_ag(df_train,
                                                       train_transfer_style,
                                                       args.poisoned_ratio,
                                                       gen_mode="update",
                                                       random_seed=seed)

            else:
                # Target Class: World
                train_transfer_style = pd.read_csv('StyleBkd/AG/transfer/train.tsv', sep='\t')
                test_transfer_style = pd.read_csv('StyleBkd/AG/transfer/test.tsv', sep='\t')

                train_transfer_style = pd.DataFrame(
                    {'label': train_transfer_style['label'], 'text': train_transfer_style['sentence']})
                test_transfer_style = pd.DataFrame(
                    {'label': test_transfer_style['label'], 'text': test_transfer_style['sentence']})

                df_train_poisoned = style_injection_ag(df_train,
                                                       train_transfer_style,
                                                       args.poisoned_ratio,
                                                       gen_mode="train",
                                                       random_seed=seed)
                df_train_poisoned.to_csv('StyleBkd/AG/train_poisoned.csv', index=False)

                df_test1_poisoned = style_injection_ag(df_test,
                                                       test_transfer_style,
                                                       args.poisoned_ratio,
                                                       gen_mode="test_part",
                                                       random_seed=seed)
                df_test1_poisoned.to_csv('StyleBkd/AG/test_poisoned_part.csv', index=False)

                df_test2_poisoned = style_injection_ag(df_test,
                                                       test_transfer_style,
                                                       args.poisoned_ratio,
                                                       gen_mode="test_all",
                                                       random_seed=seed)
                df_test2_poisoned.to_csv('StyleBkd/AG/test_poisoned_all.csv', index=False)

        else:
            raise ValueError("Invalid dataset")

    else:
        raise ValueError("Invalid injection mode")


