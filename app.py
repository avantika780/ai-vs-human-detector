import streamlit as st
from backend import predict, get_top_words

st.title("AI vs Human Text Detector")

user_input = st.text_area("Enter your text:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Enter some text")
    else:
        lr, nb = predict(user_input)

        st.subheader("Logistic Regression")
        st.write(f"{'AI Generated' if lr[0]==1 else 'Human Written'} ({lr[1]*100:.2f}%)")
        st.progress(int(lr[1]*100))

        st.subheader("Naive Bayes")
        st.write(f"{'AI Generated' if nb[0]==1 else 'Human Written'} ({nb[1]*100:.2f}%)")

if st.checkbox("Show Important Words"):
    top_ai, top_human = get_top_words()

    st.subheader("Top AI Words")
    for score, word in top_ai:
        st.write(f"{word} ({score:.2f})")

    st.subheader("Top Human Words")
    for score, word in top_human:
        st.write(f"{word} ({score:.2f})")