from textblob import TextBlob
import pandas as pd
import streamlit as st
import cleantext


st.header('Text Analysis for Positive, Neutral or Negative')
with st.expander('Write anything for Text Analytics'):
    text = st.text_input('Text here: ')
    if text:
        blob = TextBlob(text)
        st.write('Polarity: ', round(blob.sentiment.polarity,2))
        st.write('Subjectivity: ', round(blob.sentiment.subjectivity,2))


    pre = st.text_input('Clean Text here: ')
    if pre:
        st.write(cleantext.clean(pre, clean_all= False, extra_spaces=True ,
                                 stopwords=True ,lowercase=True ,numbers=True , punct=True))

with st.expander('Analyze CSV files and label your column to be analyzed as Text'):
    upl = st.file_uploader('Upload csv file here')

    def score(x):
        blob1 = TextBlob(x)
        return blob1.sentiment.polarity

#
    def analyze(x):
        if x >= 0.5:
            return 'Positive'
        elif x <= -0.5:
            return 'Negative'
        else:
            return 'Neutral'

#
    if upl:
        df = pd.read_csv(upl)
        df['score'] = df['Text'].apply(score)
        df['analysis'] = df['score'].apply(analyze)
        st.write(df.head(30))

        @st.cache
        def convert_df(df):
           
            return df.to_csv().encode('utf-8')

        csv = convert_df(df)

        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='result.csv',
            mime='text/csv',
        )