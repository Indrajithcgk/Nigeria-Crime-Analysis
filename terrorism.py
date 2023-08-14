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



st.set_page_config( page_title="Omdena  Analysis Of Crime In Nigeria",
                    page_icon='figures/logo.png',
                    layout='wide')

col1, col2 = st.columns((.5, 2))

with col1:
    logo = Image.open('figures/logo.png')
    st.image(logo)
    

with col2:
    st.title(':green[Analyzing Crime Incidents in Nigeria]')
    st.subheader("Local Chapter Enugu, Nigeria Chapter")


selected = option_menu(menu_title=None,options=['Home','Analysis','Map','Team'], 
icons=['house-fill','bar-chart-fill','globe-central-south-asia','person-fill'],orientation='horizontal',)
#Model -'x-diamond-fill'
if selected == 'Home':
    st.header('_The Problem_')
    st.markdown(""":green[Predicting Terrorist Attacks and Analyzing Crime Incidents in Nigeria Using Machine Learning] :
                    The problem this project is targeted to solve is to help the security agencies to mitigate the rate of
                    crime committed in the country by giving the security agencies reasonable insight into the distribution
                    of crime committed in Nigeria, and also enable them to anticipate possible crime and location of the 
                    crime, in order to be able to make adequate security checks and take the necessary security measures.
                    Nigeria is a country with a high level of crime.
                    The government is working to tackle the crime problem by analyzing the crimes committed 
                    and building a predictive model to predict future crimes.The goal of the project is to
                    analyze the crimes committed in Nigeria and build a dashboard to understand crime""")
    
    
    
    st.markdown(""" For further details click here
                
    [GitHub repository]("https://github.com/OmdenaAI/enugu-nigeria-crime-incidents/tree/main")
""")


    

elif selected == 'Analysis':
    #Read dataset
    def load_data():
        df = pd.read_csv("data/terrorism.csv")

        return df

    df = load_data()

    # Convert df['date'] to datetime datatype
    df['date'] = df['date'].astype('datetime64[ns]')

    #1
    st.title(":red[Crime Analysis]")
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


    col3, col4 = st.columns((1, 1.2))

    with col3:
        st.subheader("Number of Attacks changes over Years")
        fig1 = px.line(attacks_per_year, x='year', y='count', markers=True,
        labels={'year': 'Year', 'count': 'Number of Attacks'})
        st.plotly_chart(fig1,use_container_width=True)

    with col4:
        st.subheader("Number of Attacks changes over Months")
        fig2 = px.bar(attacks_per_month, x='month', y='count', color='count',width = 400,
                labels={'month': 'Month', 'count': 'Number of Attacks'}, color_continuous_scale='viridis')
        st.plotly_chart(fig2,use_container_width=True)

    #3
    #distribution of attack types
    attack_type_counts = df['attack_type'].value_counts()
    #4
    #common weapon type
    weapon_type_counts = df['weapon_type'].value_counts()


    col5,col6 = st.columns((1,1))

    with col5:
        st.subheader("Distribution of Attack Type")
        fig3 = px.pie(attack_type_counts, values=attack_type_counts.values[:6], names =attack_type_counts.index[:6],
                template="plotly_dark")
        fig3.update_traces(text  = attack_type_counts.index[:6],textposition = "inside")
        st.plotly_chart(fig3, use_container_width=True)

    with col6:
        st.subheader("Distribution of Weapon Type")
        fig4 = px.pie(weapon_type_counts, values=weapon_type_counts.values, names =weapon_type_counts.index,
                template="plotly_dark",hole=0.2)
        fig4.update_traces(text  = weapon_type_counts.index,textposition = "inside")
        st.plotly_chart(fig4, use_container_width=True)
        

    #5
    #casualties trend
    st.subheader("Trend of Casualties (Killed and Wounded) Over Time")
    casualties = df.groupby('year')[['no_killed', 'no_wounded']].sum().reset_index()
    fig5 = px.line(casualties, x='year', y=['no_killed', 'no_wounded'],
                labels={'year': 'Year', 'value': 'Number of Casualties'})
    st.plotly_chart(fig5,use_container_width=True)
            



    
    
elif selected == 'Map':
    def load_data():
        df = pd.read_csv("data/terrorism.csv")

        return df

    df = load_data()

    
        
    def create_folium_map(data, n):
        city_crime_data = data[['city', 'latitude', 'longitude']].dropna()

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
        st.subheader(':red[" Top N Cities with the highest crime density"]')
        st.write("The density or heat of crimes happening in each city using")

        # Create a slider 
        n = st.slider("Select Top N Cities:", min_value=1, max_value=100, value=5)

        # Create and display the Folium map with HeatMap for density of crimes and custom markers for top N cities
        st.header(f'Top {n} Cities with the Highest Crime Density')
        folium_map = create_folium_map(df, n)
        folium_static(folium_map) 

    if __name__ == '__main__':
        main()
      

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
        * [Devyash Jain](https://www.linkedin.com/in/devyash-jain-4abb2322a/)
        * [Indrajith C](https://www.linkedin.com/in/indrajith-c/)
        * [Melat Kebede ](https://www.linkedin.com/in/melat-kebede-291309203)
        * [Miho Rosenberg](http://www.linkedin.com/in/miho-r-93a00321a)
        * [Oluchukwu Chiamaka Okorie](https://www.linkedin.com/mwlite/in/oluchukwu-okorie-1872ba17a)
        * [Oni Samson Abidemi](https://www.linkedin.com/in/samson-oni-9bb1b7139/)
        * [Umesh Patil](https://www.linkedin.com/in/patil-umesh/)


    """
        )





       
       
       
       
