import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as pyo
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

#map
import folium
from folium.plugins import HeatMap, MarkerCluster
from streamlit_folium import folium_static

#model
import holidays
from datetime import datetime
import xgboost as xgb
import joblib

st.set_page_config( page_title="Omdena  Analysis Of Crime In Nigeria",
                    page_icon='figures/Omdena Enugu Nigeria Logo.png',
                    layout='wide')

col1, col2 = st.columns((.5, 2))

with col1:
    logo = Image.open('figures/Omdena Enugu Nigeria Logo.png')
    st.image(logo)
    

with col2:
    st.title(':green[Predicting Terrorist Attacks and Analyzing Crime Incidents in Nigeria ]')
    st.subheader("_Omdena, Enugu Nigeria Chapter, Nigeria Chapter_")
 

selected = option_menu(menu_title=None,options=['Home','Analysis','Map','Prediction','Team'], 
icons=['house-fill','bar-chart-fill','globe-central-south-asia','x-diamond-fill','person-fill'],orientation='horizontal',)

#Read dataset
def load_data():
    df = pd.read_csv("data/terrorism.csv")

    return df

df = load_data()



if selected == 'Home':


    st.write(" ")
    st.header(':blue[Project background]')

    st.write("""
                Nigeria has been identified as one of the least peaceful countries in the world, ranking 17th 
                in terms of crime rates. The first half of 2022 witnessed approximately 6,000 deaths resulting
                from various factors such as jihadist attacks, kidnappings, banditry, and military actions.
                The country's security situation necessitates addressing the issue of crime rates, which requires
                security agencies to have a comprehensive understanding of different types of crimes and the ability
                to anticipate potential outbreaks. Nigeria can effectively address its security concerns by taking 
                these measures.
             
             """)
    
    st.header(':blue[The Problem]')

    st.write("""
                The primary objective of this project is to provide valuable insights to security agencies 
                in Nigeria with the aim of reducing the high incidence of crime in the country.By analyzing past 
                criminal activities and creating a predictive model,the government can anticipate potential
                crimes and their locations, enabling them to take appropriate security measures.\n
             
                :green[The project focus is to analyze crimes committed across Nigeria and
                Predict the likelihood of an attack occurring on 
                a specific date and in a particular state.] This will aid the government in addressing the issue of 
                crime and improving the safety of citizens.
             
             """)
    

    
    

    st.write(" ")
    st.markdown(""" For further details click here
                             
    [Omdena Project Information]("https://omdena.com/chapter-challenges/analysis-and-prediction-of-crime-in-nigeria/")
""")


    

elif selected == 'Analysis':

    # Convert df['date'] to datetime datatype
    df['date'] = df['date'].astype('datetime64[ns]')



    #1
    st.title(":blue[Crime Analysis]")
    # Extract year from the 'date' column
    df['year'] = df['date'].dt.year
    attacks_per_year = df.groupby('year').size().reset_index(name='count')

    #2
    #Extract month from 'date' column
    df['month'] = df['date'].dt.month
    attacks_per_month = df.groupby('month').size().reset_index(name='count')
    # Map month numbers to month names for better readability
    month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
    attacks_per_month['month'] = attacks_per_month['month'].map(month_names)
    attacks_per_month = attacks_per_month.sort_values(by='count',ascending=True)


    col3, col4 = st.columns((1.2,1.1))

    with col3:
        st.subheader(":green[Number of Attacks changes over Years]")
        fig1 = px.line(attacks_per_year, x='year', y='count', markers=True,
        labels={'year': 'Year', 'count': 'Number of Attacks'})
        st.plotly_chart(fig1,use_container_width=True)

    with col4:
        st.subheader(":green[Number of Attacks changes over Months]")
        fig2 = px.bar(attacks_per_month, x='month', y='count', color='count',width = 400,
                labels={'month': 'Month', 'count': 'Number of Attacks'}, color_continuous_scale='Reds')
        st.plotly_chart(fig2,use_container_width=True)


    #3
    col5, col6 = st.columns((1,1))

    with col5:
        st.subheader(':green[Top 10 Cities/States having Highest Attacks]')

        # Add a filter for selecting a city or state
        location_type = st.radio("Select Location Type:", ['City', 'State'])
        

        if location_type == 'City':
            location_column = 'city'
            location_label = 'City'
        else:
            location_column = 'state'
            location_label = 'State'

        df[location_column] = df[location_column].replace('Unknown', df[location_column].mode().iloc[0])

        # Group data by location and count attacks
        location_attack_counts = df.groupby(location_column).size().reset_index(name='Attack Count')

        # Get the top 10 locations by attack count
        top_10_locations = location_attack_counts.nlargest(10, 'Attack Count')

        # Create a bar chart using Plotly for the top 10 locations
        fig3 = px.bar(top_10_locations, x=location_column, y='Attack Count',color='Attack Count',
                      color_continuous_scale="OrRd")
        st.plotly_chart(fig3)
    #4
    with col6:
        st.subheader(':green[Top 6 States - Suicide Attacks Distribution]')

        # Filter data to include only suicide attacks
        suicide_data = df[df['suicide'] == 1]

        # Group data by state and count suicide attacks
        suicide_state_counts = suicide_data.groupby('state').size().reset_index(name='suicide_count')

        # Get the top 7 states by suicide attack count
        top_6_suicide_states = suicide_state_counts.nlargest(6, 'suicide_count')

        # Create a pie chart using Plotly for top 5 states' suicide attacks distribution
        fig4 = px.pie(top_6_suicide_states, values='suicide_count', names='state',template="plotly_dark")
        fig4.update_traces(textposition='inside')

        # Customize the legend
        fig4.update_layout(legend=dict(orientation="h", x=.01, y=1.2))
        fig4.update_layout(
        margin=dict(l=90, r=20, t=30, b=0),  # Adjust margins for position
        width=440, height=500,  # Set the width and height of the chart
        )
        st.plotly_chart(fig4)















    #5
    #distribution of attack types
    attack_type_counts = df['attack_type'].value_counts()
    #6
    #common weapon type
    weapon_type_counts = df['weapon_type'].value_counts()


    col7,col8 = st.columns((1,1))

    custom_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    with col7:
        
        st.subheader(":green[Distribution of Attack Type]")
        fig5 = px.pie(attack_type_counts, values=attack_type_counts.values[:6], names =attack_type_counts.index[:6],
                template="plotly_dark")
        fig5.update_traces(text  = attack_type_counts.index[:6],textposition = "inside")
        st.plotly_chart(fig5, use_container_width=True)

    with col8:
        st.subheader(":green[Distribution of Weapon Type]")
        fig6 = px.pie(weapon_type_counts, values=weapon_type_counts.values, names =weapon_type_counts.index,
                template="plotly_dark",hole=0.2)
        fig6.update_traces(text  = weapon_type_counts.index,textposition = "inside")
        st.plotly_chart(fig6, use_container_width=True)
        




    #7
    #casualties trend
    st.subheader(":green[Trend of Casualties (Killed and Wounded) Over Time]")
    casualties = df.groupby('year')[['no_killed', 'no_wounded']].sum().reset_index()
    fig5 = px.line(casualties, x='year', y=['no_killed', 'no_wounded'],
                labels={'year': 'Year', 'value': 'Number of Casualties'},color_discrete_sequence=custom_colors)
    st.plotly_chart(fig5,use_container_width=True)
            



    
    
elif selected == 'Map':
    def load_data():
        df = pd.read_csv("data/terrorism.csv")

        return df

    df = load_data()



    def create_folium_map(data, n):
        #remove Unknown state column from the data
        data = data[data['state'] != 'Unknown']
        city_crime_data = data[['year','city', 'latitude', 'longitude']].dropna()

        # Create a  map centered on Nigeria
        map_nigeria = folium.Map(location=[9.0820, 8.6753], zoom_start=6)

        # HeatMap using the latitude and longitude of crimes in each city
        heat_data = [[row['latitude'], row['longitude']] for idx, row in city_crime_data.iterrows()]
        HeatMap(heat_data).add_to(map_nigeria)

        # Find the top N cities with the highest crime density
        top_cities = city_crime_data['city'].value_counts().nlargest(n).index.tolist()

        # Add custom markers for the top N cities showing the total sum of crimes
        for city, city_data in city_crime_data.groupby('city'):
            if city in top_cities:
                total_crimes = city_data.shape[0]
                folium.Marker(
                    location=[city_data.iloc[0]['latitude'], city_data.iloc[0]['longitude']],
                    popup=f"{city}: Total Crimes: {total_crimes}",
                    icon=folium.DivIcon(html=f"<div>Total Crimes: {total_crimes}</div>")
                ).add_to(map_nigeria)

        return map_nigeria
    


    
    def main():
        st.subheader(':blue[Top N Cities with the highest crime density]')
        st.write(":green[The density or heat of crimes happening in each city using]")

        # Create a slider 
        n = st.slider(":red[Select Top N Cities:]", min_value=1, max_value=100, value=5)

        # Create and display the Folium map with HeatMap for density of crimes and custom markers for top N cities
        st.header(f'_Top {n} Cities with the Highest Crime Density_')
        folium_map = create_folium_map(df, n)
        folium_static(folium_map) 

    if __name__ == '__main__':
        main()




elif selected == 'Prediction':
    #Load the holidays for Nigeria
    nigeria_holidays = holidays.Nigeria() 

    #Loading XGBoost model and ColumnTransformer
    model_dir = 'artifacts/'
    data_dir = 'data/'
    model_XGB_SW = xgb.XGBClassifier()
    model_XGB_SW.load_model(model_dir + 'model_XGB_SW.model')
    column_transformer = joblib.load(model_dir + 'col_transformer.pkl')

    # Loading the transformed socio-demographic data
    socio_demo_data = pd.read_csv(data_dir + 'socio_demo_for_model_prediction.csv')

    #To get the date-related information
    def nigeria_day_info(date_string):
        date_obj = datetime.strptime(date_string, '%Y-%m-%d').date()
        day_of_month = date_obj.strftime('%d')
        month = date_obj.strftime('%m')

        is_holiday = 1 if date_obj in nigeria_holidays else 0
        is_weekday = 1 if date_obj.weekday() < 5 else 0

        return month, day_of_month, is_weekday, is_holiday

    def predict_attack_prob(state, date_to_check):
        X_input = socio_demo_data.query("State == @state").copy()

        # Adding the date related info
        X_input['month'], X_input['day'], X_input['isweekday'], X_input['is_holiday'] = nigeria_day_info(date_to_check)

        # Droping unnecessary columns
        X_input.drop('Unnamed: 0', axis=1, inplace=True)

        # Column transform the input data
        X_input_transformed = column_transformer.transform(X_input)

        #predictions 
        pred_prob = np.round(model_XGB_SW.predict_proba(X_input_transformed)[0, 1] * 100, 1)

        return pred_prob

    
    st.title(":blue[Probability Of Attack Prediction]")
    st.write('_Enter the State and Date to predict the probability of an attack_')

    state = st.selectbox(':green[Select State:]', socio_demo_data['State'].unique())
    date_to_check = st.date_input(':green[Select Date:]')
    st.write(" ")
    prediction_button = st.button('Predict Probability')

    if prediction_button:
        prediction = predict_attack_prob(state, date_to_check.strftime('%Y-%m-%d'))
        prediction_color = 'red' if prediction > 40 else 'green' 
        date_state_color = "green"
        #st.write(f'Probability of an attack in :{date_state_color}[**{state}**] on :{date_state_color}[**{date_to_check.strftime("%Y-%m-%d")}**] = :{prediction_color}[**{prediction}**%]')
        prediction_size = "24px"  # Change the size as desired
        sentence_size = "28px"  # Change the size for the sentence
        st.markdown(
        f'<span style="font-size:{sentence_size};">Probability of an attack in </span>'
        f'<span style="color:{date_state_color}; font-size:{prediction_size};">{state}</span> on '
        f'<span style="color:{date_state_color}; font-size:{prediction_size};">{date_to_check.strftime("%Y-%m-%d")}</span> = '
        f'<span style="color:{prediction_color}; font-size:{prediction_size};">{prediction}%</span>',
        unsafe_allow_html=True
        )

        



elif selected == "Team":
    st.markdown(
    """
    #####
    #####    
    ### Team Lead  : 
    [Obinna (Samson) Nwachukwu](https://www.linkedin.com/in/obinna-nwachukwu-48b3881b0/)



    """)
        
    st.markdown(
        """
        ####
        ###  Project Contributors:
        * [Abomaye Victor](https://www.linkedin.com/in/abomaye-eniatorudabo-a40175107)
        * [Alamin Musa Magaga](https://www.linkedin.com/in/alamin-magaga-8b388118b/)
        * [Anjali Dashora](https://www.linkedin.com/in/anjali-dashora-4a232b204)
        * [Barakat Akinsiku](https://www.linkedin.com/in/)
        * [Danish Mehmood](https://www.linkedin.com/in/danish2014/)
        * [Devyash Jain](https://www.linkedin.com/in/devyash-jain-4abb2322a/)
        * [Ibrahim Ahmad Ismail](https://www.linkedin.com/in/ahmad-ibrahim-ismail-238a21138/)
        * [Indrajith C](https://www.linkedin.com/in/indrajith-c/)
        * [Marwan Ashraf](https://www.linkedin.com/in/marwan-ashraf-b02538195/)
        * [Melat Kebede ](https://www.linkedin.com/in/melat-kebede-291309203)
        * [Miho Rosenberg](http://www.linkedin.com/in/miho-r-93a00321a)
        * [Milind Shende](https://www.linkedin.com/in/milind-shende/)
        * [Oluchukwu Chiamaka Okorie](https://www.linkedin.com/mwlite/in/oluchukwu-okorie-1872ba17a)
        * [Oni Samson Abidemi](https://www.linkedin.com/in/samson-oni-9bb1b7139/)
        * [Robson Serafim](https://www.linkedin.com/in/robson-castro-serafim/)
        * [Rukshar Alam](https://www.linkedin.com/in/ruksharalam/)
        * [Toyyib Ogunremi](https://www.linkedin.com/in/t-ogunremi)
        * [Umesh Patil](https://www.linkedin.com/in/patil-umesh/)
        
        


    """
        )







       
       
       
       
