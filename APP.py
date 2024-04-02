import streamlit as st
import pandas as pd
import joblib

# Load precomputed data
cosine_sim = joblib.load('COSINE_SIMILARITY.pkl')
ds = joblib.load('DATA_ITEMS.pkl')

# Function to recommend places
def tourism_recommendations(place_name):
    try:
        similarity_data = cosine_sim
        items = ds[['name', 'category', 'description', 'city']]
    
        k = 5
        index = similarity_data.loc[:, place_name].to_numpy().argsort()[::-1][:k]
        closest = similarity_data.columns[index[-1:-(k+2):-1]]
        closest = closest.drop(place_name, errors='ignore')
        return pd.DataFrame(closest).merge(items).head(k)
    except Exception as e:
       pass

# Streamlit app
def main():
    st.title('Travel Recommendation System')
    st.subheader('Enter the location below to see recommendations')

    # Input field for location
    place = st.text_input("Enter your location here")

    # Button to generate recommendations
    recommend_button = st.button("Generate Recommendations")

    # Display recommendations if button clicked
    if recommend_button and place:
        recommendations = tourism_recommendations(place)
        if recommendations is not None and not recommendations.empty:
            st.write(recommendations)
        else:
            st.warning("Sorry! No recommendations found.")

if __name__ == '__main__':
    main()