from io import StringIO
import boto3
from boto3 import session
import os
import platform
import json
import pandas as pd
import numpy as np
from simpletransformers.data.url_vocab import UrlVocab


PLATFORM = platform.system()
if PLATFORM != 'Darwin':
    sts_client = boto3.client('sts', region_name='us-west-2')
    databricks_session = boto3.DEFAULT_SESSION


def get_resource():
    return boto3.resource('s3', region_name='us-west-2')


def get_csv(s3, bucket, file_path):
    obj = s3.Object(bucket, file_path)
    return pd.read_csv(obj.get()['Body'])


def get_json(s3, bucket, file_path):
    obj = s3.Object(bucket, file_path)
    return json.load(obj.get()['Body'])


def get_outreach_session(sts_client):
    """Specifically for writing data from Databricks to S3."""
    assumed_role_object = sts_client.assume_role(
        RoleArn="arn:aws:iam::182192988802:role/data-science-databricks-assume-role",
        RoleSessionName="AssumeRoleSession1")
    credentials = assumed_role_object['Credentials']
    outreach_session = session.Session(aws_access_key_id=credentials['AccessKeyId'],
                                       aws_secret_access_key=credentials['SecretAccessKey'],
                                       aws_session_token=credentials['SessionToken'])
    print(outreach_session._session._register_credential_provider)
    return outreach_session


def write_to_s3(name, df):
    """write to s3 with outreach session"""
    if PLATFORM == 'Darwin':
        # Local Mac
        csv_buffer = StringIO()
        if isinstance(df, pd.DataFrame):
            csv_buffer.write(df.to_csv(index=False))
        else:
            csv_buffer.write(json.dumps(df))
        s3.Object(BUCKET, name).put(Body=csv_buffer.getvalue())
    else:
        # Databricks
        s3_outreach = get_outreach_session(sts_client).resource('s3')
        csv_buffer = StringIO()
        if isinstance(df, pd.DataFrame):
            csv_buffer.write(df.to_csv(index=False))
        else:
            csv_buffer.write(json.dumps(df))
        s3_outreach.Object(BUCKET, name).put(Body=csv_buffer.getvalue())


s3 = get_resource()
BUCKET = 'annotation-databricks-access-granted'


def load_url_vocab(url_file_path,
                   node2vec_encode=False,
                   bert_encode=False,
                   bert_encoding_filename=None,
                   bert_encode_email_context=False,
                   bert_encoding_email_filename=None):
    # Load URL metadata
    jsonstring = get_json(s3, BUCKET, url_file_path)
    df_url = pd.io.json.json_normalize(jsonstring)

    # Remove some URLs from the predication candidates
    df_url = df_url[(~df_url['header'].isnull()) & (~df_url['article'].isnull())]
    df_url = df_url[df_url['header'].apply(lambda x: len(x) > 0) & df_url['article'].apply(lambda x: len(x) > 0)]
    df_url = df_url[df_url['nav_hrefs'].apply(lambda x: len(x) > 0)]

    # Remove surrounding white spaces in text fields
    df_url['header'] = df_url['header'].apply(lambda x: x.strip())

    urlvocab = UrlVocab(df_url,
                        node2vec_encode=node2vec_encode,
                        bert_encode=bert_encode,
                        bert_encoding_filename=bert_encoding_filename,
                        bert_encode_email_context=bert_encode_email_context,
                        bert_encoding_email_filename=bert_encoding_email_filename)
    return urlvocab


def load_url_data_email_only(datafolder, urlvocab):
    def get_split(datafolder, split):
        df = get_csv(s3, BUCKET, os.path.join(datafolder, split))
        df['labels'] = df['labels'].apply(json.loads)
        df['prospect_reply_message'] = df['prospect_reply_message'].apply(lambda x: x.lower())
        df = df.rename(columns={'prospect_reply_message': 'text', 'labels': 'url_labels'})
        df['labels'] = df['url_labels'].apply(lambda x: urlvocab.encode_urls(x))
        return df

    df_train = get_split(datafolder, 'train.csv')
    df_dev = get_split(datafolder, 'dev.csv')
    df_test = get_split(datafolder, 'test.csv')
    return df_train, df_dev, df_test


def load_url_data_email_article_pair(datafolder, urlvocab, onlytitle=False, include_url_id=False):
    '''
    Getting URL prediction datasets. "text_a" field contains input emails and "text_b" field contains the article
    associated with the candidate URL.
    :param datafolder:
    :param urlvocab:
    :return:
    '''
    def get_split(datafolder, split):
        df = get_csv(s3, BUCKET, os.path.join(datafolder, split))
        df['labels'] = df['labels'].apply(json.loads)
        df['prospect_reply_message'] = df['prospect_reply_message']
        df = df.rename(columns={'prospect_reply_message': 'text_a', 'labels': 'url_labels'})

        # Get (url, label) lists for each input example
        df['url_label_lists'] = df['url_labels'].apply(
            lambda x: list(zip(urlvocab.url_vocab_list, urlvocab.encode_urls(x))))

        # Expand url_label lists
        # The purpose is to transfer the problem to binary classificaiton.
        #
        # For example, there are five urls in total [A,B,C,D,E]
        # For a sampple (df row) with "url_label" = [A,D]
        # We will expand this sample into 5, where their ("url", "label") are
        # (A,1), (B,0), (C,0), (D,1), (E,0) respectively.
        s_q = df.apply(lambda x: pd.Series(x['url_label_lists']), axis=1).stack().reset_index(level=1, drop=True)
        s_q.name = 'url_label'
        df_expand_label = df.copy(deep=True)
        df_expand_label = df_expand_label.drop('url_label_lists', axis=1).join(s_q)
        assert len(df_expand_label) == sum(df['url_label_lists'].apply(len))

        df_expand_label['url'] = df_expand_label['url_label'].apply(lambda x: x[0])
        df_expand_label['labels'] = df_expand_label['url_label'].apply(lambda x: x[1])
        df_expand_label.drop('url_label', axis=1, inplace=True)

        # Get context information
        if onlytitle:
            df_expand_label['text_b'] = df_expand_label['url'].apply(
                lambda x: urlvocab.get_title(x))
        else:
            df_expand_label['text_b'] = df_expand_label['url'].apply(
                lambda x: urlvocab.get_title(x) + '. ' + urlvocab.get_text(x))

        if include_url_id:
            df_expand_label['url_id'] = df_expand_label['url'].apply(
                lambda x: urlvocab.url2idx(x))

        return df_expand_label

    df_train = get_split(datafolder, 'train.csv')
    df_dev = get_split(datafolder, 'dev.csv')
    df_test = get_split(datafolder, 'test.csv')
    return df_train, df_dev, df_test


def get_feature_connectivity(url, urlvocab):
    return urlvocab.url2nav(url) + urlvocab.url2outconn(url) + urlvocab.url2inconn(url)


def get_feature_node2vec(url, urlvocab):
    return urlvocab.url2node2vec(url).tolist()


def get_feature_bert(url, urlvocab):
    return urlvocab.url2bert(url).tolist()


def get_feature_email_context(url, urlvocab):
    return urlvocab.url2emailcontextembedding(url).tolist()


def load_url_data_with_neighbouring_info(datafolder, urlvocab, onlytitle=False,
                                         addfeatures=None,
                                         include_url_id=False):
    '''
    Getting URL prediction datasets. "text_a" field contains input emails and "text_b" field contains the article
    associated with the candidate URL.
    :param datafolder:
    :param urlvocab:
    :param onlytitle:
    :param addfeatures:
    :return:
    '''
    def get_split(datafolder, split):
        df = get_csv(s3, BUCKET, os.path.join(datafolder, split))
        df['labels'] = df['labels'].apply(json.loads)
        df['prospect_reply_message'] = df['prospect_reply_message']
        df = df.rename(columns={'prospect_reply_message': 'text_a', 'labels': 'url_labels'})

        # Get (url, label) lists for each input example
        df['url_label_lists'] = df['url_labels'].apply(
            lambda x: list(zip(urlvocab.url_vocab_list, urlvocab.encode_urls(x))))

        # Expand url_label lists
        # The purpose is to transfer the problem to binary classificaiton.
        #
        # For example, there are five urls in total [A,B,C,D,E]
        # For a sampple (df row) with "url_label" = [A,D]
        # We will expand this sample into 5, where their ("url", "label") are
        # (A,1), (B,0), (C,0), (D,1), (E,0) respectively.
        s_q = df.apply(lambda x: pd.Series(x['url_label_lists']), axis=1).stack().reset_index(level=1, drop=True)
        s_q.name = 'url_label'
        df_expand_label = df.copy(deep=True)
        df_expand_label = df_expand_label.drop('url_label_lists', axis=1).join(s_q)
        assert len(df_expand_label) == sum(df['url_label_lists'].apply(len))

        df_expand_label['url'] = df_expand_label['url_label'].apply(lambda x: x[0])
        df_expand_label['labels'] = df_expand_label['url_label'].apply(lambda x: x[1])
        df_expand_label.drop('url_label', axis=1, inplace=True)

        # Get context information
        # FIXME: another option is to load this URL-related data during model training
        if onlytitle:
            df_expand_label['text_b'] = df_expand_label['url'].apply(
                lambda x: urlvocab.get_title(x))
        else:
            df_expand_label['text_b'] = df_expand_label['url'].apply(
                lambda x: urlvocab.get_title(x) + '. ' + urlvocab.get_text(x))

        # Get other information as additional features
        # FIXME: another option is to load this URL-related data during model training
        if addfeatures == 'connectivity':
            df_expand_label['addfeatures'] = df_expand_label['url'].apply(
                lambda x: get_feature_connectivity(x, urlvocab))
        elif addfeatures == 'node2vec':
            df_expand_label['addfeatures'] = df_expand_label['url'].apply(
                lambda x: get_feature_node2vec(x, urlvocab))
        elif addfeatures == 'bert':
            df_expand_label['addfeatures'] = df_expand_label['url'].apply(
                lambda x: get_feature_bert(x, urlvocab))
        elif addfeatures == 'email_context':
            df_expand_label['addfeatures'] = df_expand_label['url'].apply(
                lambda x: get_feature_email_context(x, urlvocab))

        if include_url_id:
            df_expand_label['url_id'] = df_expand_label['url'].apply(
                lambda x: urlvocab.url2idx(x))

        return df_expand_label


    df_train = get_split(datafolder, 'train.csv')
    df_dev = get_split(datafolder, 'dev.csv')
    df_test = get_split(datafolder, 'test.csv')
    return df_train, df_dev, df_test


if __name__ == '__main__':
    file_path = os.path.join('jie-faq', 'faq', 'outreach_support_url_meta.json')
    urlvocab = load_url_vocab(file_path)


