import streamlit as st
import pandas as pd 
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import pickle
import numpy as np
from datetime import datetime

st.set_page_config(
    page_title="Airline Passenger Satisfaction Analytics",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded")


# *Prompt an ChatGPT:
# 1. Ich möchte ein einheitliches Dark-Blue-Design (HEX: #003566) für alle Texte und Diagrammachsen in Streamlit anwenden – wie geht das?

# Apply custom styling
st.markdown("""
<style>
/* Globale Textfarbe */
html, body, [class*="css"] {
    color: #003566;
}

/* Titel & Header */
h1, h2, h3, h4, h5, h6, .stMarkdown, .stText, .stSubheader, .stHeader {
    color: #003566 !important;
}

/* Sidebar Texte */
.css-1d391kg, .css-qrbaxs, .stRadio, .stSelectbox, .stMultiSelect {
    color: #003566 !important;
}

/* Diagramm-Titel optional überschreiben */
.plot-container .xtick, .ytick, .xaxis-title, .yaxis-title, .legendtext {
    color: #003566 !important;
}
</style>
""", unsafe_allow_html=True)


# --------------- Load Data -----------------
# -------------------------------------------

# Prompts an ChatGPT:
# 1. Wie kann ich in Streamlit beim Laden einer CSV-Datei automatisch leere Spalten wie 'Unnamed: 0' oder 'id' entfernen?
# 2. Ich möchte in meiner Streamlit-App einen gecachten DataLoader bauen, der auch fehlende Werte bei 'Arrival Delay in Minutes' sinnvoll behandelt. Wie mache ich das am besten?
# 3. Wie kann ich bei einem Airline-Datensatz sicherstellen, dass alle Rating-Spalten vom Typ int sind und fehlende/nicht-numerische Werte sauber durch den Median ersetzt werden?


@st.cache_data
def load_data():
    data = pd.read_csv("capstone_merged_data.csv")
    data = data.dropna()

    # Drop the unnamed column if it exists
    if "Unnamed: 0" in data.columns:
        data = data.drop(columns=["Unnamed: 0"])
    if "id" in data.columns:
        data = data.drop(columns=["id"])

    # Handle missing values for 'Arrival Delay in Minutes'
    if "Arrival Delay in Minutes" in data.columns:
        median_arrival_delay = data["Arrival Delay in Minutes"].median()
        data["Arrival Delay in Minutes"].fillna(median_arrival_delay, inplace=True)
    
    # Convert satisfaction to numerical for easier calculations (0 for neutral/dissatisfied, 1 for satisfied)
    data["satisfaction_numeric"] = data["satisfaction"].apply(lambda x: 1 if x == "satisfied" else 0)
    
    # Ensure categorical columns are treated as such for Plotly
    categorical_cols = ["Gender", "Customer Type", "Type of Travel", "Class", "satisfaction"]
    for col in categorical_cols:
        if col in data.columns:
            data[col] = data[col].astype("category")
            
    # Service rating columns (assuming 0-5 or 1-5 scale)
    rating_cols = [
        "Inflight wifi service", "Departure/Arrival time convenient", "Ease of Online booking",
        "Gate location", "Food and drink", "Online boarding", "Seat comfort",
        "Inflight entertainment", "On-board service", "Leg room service", "Baggage handling",
        "Checkin service", "Inflight service", "Cleanliness"]

    for col in rating_cols:
        if col in data.columns:
            # Ensure they are numeric, handle cases where they might be objects if data is dirty
            data[col] = pd.to_numeric(data[col], errors="coerce") 
            # For simplicity, let's fill any coerced NaNs with median. A more robust approach might be needed for real-world dirty data.
            if data[col].isnull().any():
                data[col].fillna(data[col].median(), inplace=True)
            data[col] = data[col].astype(int) # Assuming ratings are integers

    return data


# ------------- Simple Model ----------------
# -------------------------------------------

# Prompts an ChatGPT:
# 1. Wie kann ich ein einfaches RandomForest-Modell in Streamlit erstellen, das nur die Eingabefelder aus dem UI nutzt?
# 2. Wie skaliere ich numerische Features mit StandardScaler und speichere gleichzeitig den Scaler für spätere Vorhersagen?

@st.cache_resource
def create_simple_model():
    """
    Creates a simple model that works with the UI features
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    
    # Get the data
    df = load_data()
    
    # Use only the features we display in the UI for simplicity
    features = [
        'Age', 'Flight Distance', 'Inflight wifi service', 
        'Departure/Arrival time convenient', 'Ease of Online booking',
        'Gate location', 'Food and drink', 'Online boarding',
        'Seat comfort', 'Inflight entertainment', 'On-board service',
        'Leg room service', 'Baggage handling', 'Checkin service',
        'Inflight service', 'Cleanliness', 'Departure Delay in Minutes']
    
    # Create X and y
    X = df[features]
    y = df['satisfaction_numeric']
    
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create and train the model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler, features


# ---------- Load Model (XGBoost) -----------
# -------------------------------------------

@st.cache_resource
def load_model():
    filename = "xgb_model.sav"
    loaded_model = pickle.load(open(filename, "rb"))
    return loaded_model


# ---------------- Main App -----------------
# -------------------------------------------

data = load_data()
model = load_model()


# ----------- Sidebar Navigation ------------
# -------------------------------------------

# Prompt an ChatGPT:
# 1. Wie kann ich in Streamlit eine einfache Seiten-Navigation erstellen?"

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Executive Dashboard", "Detailed Analysis", "Satisfaction Prediction", "Customer Journey Map", "Recommendations"])


# ---------- Executive Dashboard ------------
# -------------------------------------------

# Prompt an ChatGPT:
# 1. Wie kann ich in Streamlit Bilder auf meine App hochladen?

if page == "Executive Dashboard":
    st.title("Executive Dashboard - Passenger Satisfaction Overview")
    st.image("dashboard.jpg", use_column_width=True)

    st.markdown("""
    This **Executive Dashboard** provides a comprehensive overview of **passenger satisfaction trends** and **operational performance** within the airline.
    Use the interactive **filters** on the left to explore customer sentiment across different travel classes, customer types, and travel purposes.
    Key performance indicators (**KPIs**) at the top of the page summarize current satisfaction rates, net satisfaction score, and average delays.
    Below the KPIs, dynamic **visualizations** illustrate satisfaction distributions and **segment-specific insights**. A data-driven **feature importance analysis** highlights which service factors most influence satisfaction — helping identify **key drivers** and **areas for improvement**.
    All this information based on setted filters can be exported to a .csv file by just clicking the **Download Button**.
    """)
    

    # Sidebar Filter Section
    # ----------------------------------------

    # Prompts an ChatGPT:
    # 1. Wie kann ich in Streamlit benutzerfreundliche Filter erstellen, um Daten flexibel zu analysieren?
    # 2. Wie kann ich sicherstellen, dass bei interaktiven Filtern kein leerer DataFrame verarbeitet wird, aktuell habe ich eine Fehlermeldung?

    st.sidebar.header("Filters")
    customer_type_filter = st.sidebar.multiselect(
        "Customer Type",
        options=data["Customer Type"].unique().tolist(),
        default=data["Customer Type"].unique().tolist())

    class_filter = st.sidebar.multiselect(
        "Class",
        options=data["Class"].unique().tolist(),
        default=data["Class"].unique().tolist())

    type_of_travel_filter = st.sidebar.multiselect(
        "Type of Travel",
        options=data["Type of Travel"].unique().tolist(),
        default=data["Type of Travel"].unique().tolist())

    gender_filter = st.sidebar.multiselect(
        "Gender",
        options=data["Gender"].unique().tolist(),
        default=data["Gender"].unique().tolist())

    data_filtered = data[
        (data["Customer Type"].isin(customer_type_filter)) &
        (data["Class"].isin(class_filter)) &
        (data["Type of Travel"].isin(type_of_travel_filter)) &
        (data["Gender"].isin(gender_filter))]

    if data_filtered.empty:
        st.warning("No data matches the selected filters. Please adjust your filter criteria.")
        st.stop() # Stop execution if no data after filtering

    
    # Key Performance Indicators
    # ----------------------------------------

    # Prompts an ChatGPT:
    # 1. Wie implementiere ich dynamische KPIs in Streamlit, die sich automatisch basierend auf Filteroptionen aktualisieren?
    # 2. Muss ich HTML/CSS verwenden, um meine KPIs in Streamlit optisch ansprechend in Boxen darzustellen?
    # 3. Wie kann ich Plotly-Diagramme optisch an meine Farbe (#003566) anpassen und die Schriftarten konsistent gestalten?
    # 4. Wie muss ich meinen Code anpassen, dass Achsentitel, Legenden und Prozentangaben gut lesbar und visuell cool angezeigt werden (verwende Plotly)?

    st.markdown("---    ")
    st.subheader("Key Performance Indicators")
    overall_satisfaction_rate = data_filtered["satisfaction_numeric"].mean() * 100
    satisfied_count = data_filtered[data_filtered["satisfaction"] == "satisfied"].shape[0]
    neutral_dissatisfied_count = data_filtered[data_filtered["satisfaction"] == "neutral or dissatisfied"].shape[0]
    
    if (satisfied_count + neutral_dissatisfied_count) > 0:
        net_satisfaction_score = (satisfied_count - neutral_dissatisfied_count) / (satisfied_count + neutral_dissatisfied_count) * 100
    else:
        net_satisfaction_score = 0
        
    avg_departure_delay = data_filtered["Departure Delay in Minutes"].mean()
    
    
    loyal_satisfied_count = data_filtered[(data_filtered["Customer Type"] == "Loyal Customer") & (data_filtered["satisfaction"] == "satisfied")].shape[0]
    loyal_total_count = data_filtered[data_filtered["Customer Type"] == "Loyal Customer"].shape[0]

    if loyal_total_count > 0:
        loyalty_retention_rate = loyal_satisfied_count / loyal_total_count * 100
    else:
        loyalty_retention_rate = 0

    st.markdown("""
        <style>
        .kpi-card {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        }
        .kpi-value {
            font-size: 32px;
            font-weight: bold;
            color: #003566;
            margin-top: 10px;
        }
        .kpi-label {
            font-size: 16px;
            color: #333333;
            margin-bottom: 0;
        }
        </style>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Overall Satisfaction Rate</div>
                <div class="kpi-value">{overall_satisfaction_rate:.2f}%</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Net Satisfaction Score</div>
                <div class="kpi-value">{net_satisfaction_score:.2f}</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Avg. Departure Delay</div>
                <div class="kpi-value">{avg_departure_delay:.2f} min</div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">Loyalty Retention Rate</div>
                <div class="kpi-value">{loyalty_retention_rate:.2f}%</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    total_rows = data.shape[0]
    filtered_rows = data_filtered.shape[0]

    st.markdown(
        f"<p style='text-align:center; font-size:16px; color:gray;'>"
        f"<strong>{filtered_rows}</strong> out of <strong>{total_rows}</strong> data sets are included in the analysis "
        f"(based on your filter settings).</p>",
        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)


    # Satisfaction Charts
    # ----------------------------------------

    # Prompts an ChatGPT:
    # 1. Kannst du mir die Kundenzufriedenheit (verteilte Satisfaction) visuell in einer Art Balkendiagramm mit Plotly darstellen?
    # 2. Wie vergleiche ich verschiedene Kundensegmente visuell in Bezug auf Zufriedenheitskennzahlen mit Plotly und Streamlit?
    # 3. Kann man mittels Plotly ein Donut-Diagramm erstellen, das die Zufriedenheitsraten pro Klasse (z.B. Economy, Business) als Prozent darstellt?

    st.markdown("---    ")
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.subheader("Distribution of Satisfaction")
        satisfaction_dist = data_filtered["satisfaction"].value_counts().reset_index()
        satisfaction_dist.columns = ["satisfaction", "count"]
        fig_dist = px.bar(satisfaction_dist, x="satisfaction", y="count", color="satisfaction",
                          color_discrete_map={"satisfied": "green", "neutral or dissatisfied": "red"},
                          labels={"count": "Number of Customers", "satisfaction": "Satisfaction Level"})
        fig_dist.update_layout(showlegend=False)
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_chart2:
        st.subheader("Representation Rate by Class")
        satisfaction_by_class = data_filtered.groupby("Class")["satisfaction_numeric"].mean().reset_index()
        satisfaction_by_class.columns = ["Class", "Satisfaction Rate"]
        satisfaction_by_class["Satisfaction Rate"] *= 100 # Convert to percentage
        fig_class_donut = px.pie(satisfaction_by_class, names="Class", values="Satisfaction Rate", hole=0.4,
                                 title="", color_discrete_sequence=["#003566", "#005792", "#3399cc"])
        fig_class_donut.update_traces(textfont_color="white", textinfo="percent+label")
        st.plotly_chart(fig_class_donut, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)


    # Feature Importance Analysis
    # ----------------------------------------

    # Prompts an ChatGPT:
    # 1. Bitte gib mir einen Code, der die wichtigsten Einflussfaktoren auf Kundenzufriedenheit anhand einer Korrelationsmatrix identifizieren und sortieren (wichtigste zuoberst)?
    # 2. Wie kann ich auf den horizontalen Bar-Chart in Plotly die Resultate anzeigen lassen in weisser Farbe?

    st.markdown("---    ")
    st.subheader("Key Drivers of Satisfaction (Top 10)")

    numeric_cols = data_filtered.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col != "satisfaction_numeric"]

    correlations = data_filtered[numeric_cols + ["satisfaction_numeric"]].corr()
    feature_importance = correlations["satisfaction_numeric"].drop("satisfaction_numeric").sort_values(ascending=False)

    fig = go.Figure(data=[
        go.Bar(
            x=feature_importance.head(20).values[::-1],
            y=feature_importance.head(20).index[::-1],
            orientation='h',
            marker_color="#003566",
            text=[f"{v:.2f}" for v in feature_importance.head(10).values[::-1]],
            textposition="auto")])

    fig.update_layout(
        xaxis_title="Correlation with Satisfaction",
        yaxis_title="Feature",
        height=450,
        margin=dict(l=40, r=40, t=60, b=40),)

    st.plotly_chart(fig, use_container_width=True)


    # Export Filtered Data
    # ----------------------------------------

    # Prompts an ChatGPT:
    # 1. Gib mir einen Code, der es den Nutzern ermöglicht, die gefilterten bzw. analysierte Daten herunterzuladen in Excel Form?
    # 2. Welche Funktion verwende ich in Streamlit, um eine Datei zum Download bereitzustellen?
    # 3. Kannst du mir einen Button erstellen, der für den Nutzer interaktiv seins soll?

    st.markdown("---    ")
    st.subheader("Export Filtered Data")
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode("utf-8")

    csv = convert_df_to_csv(data_filtered)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name="filtered_airline_satisfaction_data.csv",
        mime="text/csv",)
    

# ----------- Detailed Analysis -------------
# -------------------------------------------

# Prompts an ChatGPT:
# 1. Wie kann ich in st.markdowns Bullet Points erstellen? 

elif page == "Detailed Analysis":
    st.title("Detailed Customer Analysis")
    st.image("analysis.jpg", use_column_width=True)  

    st.markdown("""
    This section provides a **deep-dive analytical view** into customer satisfaction data. 
    By using interactive filters (e.g., age, travel class, customer type, flight distance), you can:

    - **Analyze average service ratings** for satisfied vs. dissatisfied passengers using a comparative heatmap.  
    - **Examine rating distributions** for selected service factors to spot satisfaction patterns.  
    - **Evaluate the effect of delays** on satisfaction levels with a visual time-impact analysis.  
    - **Identify key service gaps**, highlighting which areas need improvement based on passenger feedback.  
    - **Export the filtered data** for offline reporting or further processing.

    This page empowers airline analysts and decision-makers to uncover actionable insights, 
    target improvement areas precisely, and support strategic customer experience enhancements.
    """)
    
    # Sidebar Filter Section
    # ----------------------------------------

    # Prompt an ChatGPT:
    # 1. Bitte passe mein Code von oben entsprechend der Seite "Detailed Analysis" an und nehme noch "Age" und "Flight Distance" hinzu, am besten mit einer Range

    st.sidebar.header("Filters")
    
    age_range = st.sidebar.slider("Age Range",
                                 int(data["Age"].min()), 
                                 int(data["Age"].max()),
                                 (int(data["Age"].min()), int(data["Age"].max())))
    
    distance_range = st.sidebar.slider("Flight Distance Range (km)",
                                      int(data["Flight Distance"].min()),
                                      int(data["Flight Distance"].max()),
                                      (int(data["Flight Distance"].min()), int(data["Flight Distance"].max())))
    
    customer_type_filter = st.sidebar.multiselect(
        "Customer Type",
        options=data["Customer Type"].unique().tolist(),
        default=data["Customer Type"].unique().tolist())

    class_filter = st.sidebar.multiselect(
        "Class",
        options=data["Class"].unique().tolist(),
        default=data["Class"].unique().tolist())

    type_of_travel_filter = st.sidebar.multiselect(
        "Type of Travel",
        options=data["Type of Travel"].unique().tolist(),
        default=data["Type of Travel"].unique().tolist())

    gender_filter = st.sidebar.multiselect(
        "Gender",
        options=data["Gender"].unique().tolist(),
        default=data["Gender"].unique().tolist())
    
    filtered_data = data[
        (data["Age"] >= age_range[0]) & (data["Age"] <= age_range[1]) & 
        (data["Flight Distance"] >= distance_range[0]) & (data["Flight Distance"] <= distance_range[1]) &
        (data["Gender"].isin(gender_filter)) &
        (data["Customer Type"].isin(customer_type_filter)) &
        (data["Class"].isin(class_filter)) &
        (data["Type of Travel"].isin(type_of_travel_filter))]

    
    # Show Amount of Filtered Results
    # ----------------------------------------

    # Prompts an ChatGPT:
    # 1. Wie kann ich anzeigen lassen, wie viele Zeilen bzw. Einträge mit der entsprechenden Filterung von den Total miteinbezogen werden?
    # 2. Berechne die Differenz zwischen der Gesamtdatenmenge und der gefilterten Datenmenge.

    st.markdown("---    ")
    st.subheader("Filtered Results")

    total_rows = data.shape[0]
    filtered_rows = filtered_data.shape[0]

    st.markdown(
        f"<p style='text-align:center; font-size:16px; color:gray;'>"
        f"<strong>{filtered_rows}</strong> out of <strong>{total_rows}</strong> data sets are included in the analysis "
        f"(based on your filter settings).</p>",
        unsafe_allow_html=True)
    

    # Heatmap of Service Ratings
    # ----------------------------------------

    # Prompts an ChatGPT:
    # 1. Erstelle mir eine Heatmap, die die durchschnittlichen Servicebewertungen je nach Zufriedenheit anzeigt.
    # 2. Verwende Plotly, um eine farblich abgestufte Heatmap im Stil 'YlGnBu' zu erzeugen.
    # 3. Setze die Achsenbeschriftungen sowie Farbskala der Heatmap gezielt, damit der Inhalt verständlich ist.
    # 4. Wie kann ich Textwerte automatisch in der Heatmap anzeigen lassen (z. B. Rating-Werte)?

    st.markdown("---    ")
    st.subheader("Average Service Ratings by Satisfaction Level")
    st.markdown("<br>", unsafe_allow_html=True)
    service_cols = ['Inflight wifi service', 'Departure/Arrival time convenient', 
                   'Ease of Online booking', 'Gate location', 'Food and drink', 
                   'Online boarding', 'Seat comfort', 'Inflight entertainment', 
                   'On-board service', 'Leg room service', 'Baggage handling', 
                   'Checkin service', 'Inflight service', 'Cleanliness']
    
    # Prepare data
    service_means = filtered_data.groupby('satisfaction')[service_cols].mean().T.round(1).reset_index()
    service_means = service_means.rename(columns={'index': 'Service Factor'})

    # Melt for plotly
    df_melted = service_means.melt(id_vars='Service Factor', var_name='Satisfaction', value_name='Avg Rating')
    
    # Plotly Heatmap
    fig = px.imshow(
        df_melted.pivot(index='Service Factor', columns='Satisfaction', values='Avg Rating').values,
        x=df_melted['Satisfaction'].unique(),
        y=service_means['Service Factor'],
        color_continuous_scale='YlGnBu',
        text_auto=True,
        aspect="auto")

    fig.update_layout(
        height=500,
        margin=dict(l=40, r=40, t=40, b=40),
        coloraxis_colorbar=dict(title="Avg Rating"),
        xaxis_title="Satisfaction",
        yaxis_title="Service Factor",
        font=dict(color="#003566"))

    st.plotly_chart(fig, use_container_width=True)


    # Distribution Analysis
    # ----------------------------------------

    # Prompts an ChatGPT:
    # 1. Mach zwei Balkendiagramme, die die Bewertungsverteilung für zufriedene und unzufriedene Kunden nebeneinander darstellen.
    # 2. Wie kann ich die Daten je nach Zufriedenheit filtern und getrennt zählen?
    # 3. Zeige die Balkenbeschriftungen direkt im Diagramm an.
    # 4. Verwende immer noch das Farbschema (#003566), ab nun für sämtliche Grafiken.

    st.markdown("---    ")
    st.subheader("Service Factor Distribution Analysis")

    # Let user select a service factor to analyze
    selected_service = st.selectbox("Select Service Factor", service_cols)

    satisfied_data = filtered_data[filtered_data['satisfaction'] == 'satisfied'][selected_service]
    dissatisfied_data = filtered_data[filtered_data['satisfaction'] == 'neutral or dissatisfied'][selected_service]

    satis_hist = satisfied_data.value_counts().sort_index()
    dissat_hist = dissatisfied_data.value_counts().sort_index()
    
    # Create two columns layout
    col1, col2 = st.columns(2)
    
    # Create distribution plot for satisfied
    with col1:
        fig1 = go.Figure(data=[
            go.Bar(
                x=satis_hist.index.astype(str),
                y=satis_hist.values,
                text=satis_hist.values,
                textposition="auto",
                marker_color="#003566")])

        fig1.update_layout(
            title=f"Satisfied - {selected_service}",
            title_x=0.4,
            xaxis_title="Rating",
            yaxis_title="Count",
            height=400,
            margin=dict(l=40, r=20, t=50, b=40))

        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = go.Figure(data=[
            go.Bar(
                x=dissat_hist.index.astype(str),
                y=dissat_hist.values,
                text=dissat_hist.values,
                textposition="auto",
                marker_color="#003566")])

        fig2.update_layout(
            title=f"Neutral or Dissatisfied - {selected_service}",
            title_x=0.35,
            xaxis_title="Rating",
            yaxis_title="Count",
            height=400,
            margin=dict(l=40, r=20, t=50, b=40))

        st.plotly_chart(fig2, use_container_width=True)


    # Time-Based Analysis
    # ----------------------------------------

    # Prompts an ChatGPT:
    # 1. Ich möchte gerne den Einfluss von Verspätungen "Delay" auf die Kundenzufriedenheit darstellen und möchte, dass du mir dabei hilfst das mit verschiedenen Verspätungs-Clusters zu machen
    # 2. Berechne mir die durchschnittliche Zufriedenheit pro Delay-Kategorie aus und zeig die Werte in Prozent an.
    # 3. Erstelle einen Plotly-Graph mit Balken in Farbe #003566 und Prozentwerten über jedem Balken.
    # 4. Zentriere bitte den Diagrammtitel und beschrifte die Achsen sauber (X = Delay, Y = Satisfaction Rate).

    st.markdown("---    ")
    st.subheader("Delay Impact Analysis")

    # Delay Binning
    bins = [0, 15, 30, 60, 120, filtered_data['Departure Delay in Minutes'].max()]
    labels = ['0-15 min', '15-30 min', '30-60 min', '1-2 hours', '2+ hours']
    filtered_data['delay_category'] = pd.cut(
        filtered_data['Departure Delay in Minutes'], 
        bins=bins, labels=labels, right=False)

    delay_sat_rate = (filtered_data.groupby('delay_category')['satisfaction_numeric'].mean().reset_index().rename(columns={'satisfaction_numeric': 'Satisfaction Rate'})) # Satisfaction Rate by category
    delay_sat_rate['Satisfaction Rate'] *= 100

    # Visualization
    fig = go.Figure(data=[
        go.Bar(
            x=delay_sat_rate["delay_category"],
            y=delay_sat_rate["Satisfaction Rate"],
            marker_color="#003566",
            text=delay_sat_rate["Satisfaction Rate"].apply(lambda x: f"{x:.1f}%"),
            textposition="auto")])

    fig.update_layout(
        title="Satisfaction Rate by Departure Delay",
        title_x=0.39,
        yaxis_title="Satisfaction Rate (%)",
        xaxis_title="Delay Duration",
        height=400,
        margin=dict(l=40, r=40, t=60, b=40))

    st.plotly_chart(fig, use_container_width=True)


    # Key insights section
    # ----------------------------------------

    # Prompts an ChatGPT:
    # 1. Berechne mir die allgemeine Zufriedenheitsrate basierend auf einem gefilterten Datensatz?
    # 2. Welche Service-Kategorie zeigt den grössten Unterschied zwischen zufriedenen und unzufriedenen Kunden?
    # 3. Wie gross ist der durchschnittliche Verspätungsunterschied zwischen zufriedenen und unzufriedenen Kunden?
    # 4. Stelle mir diese Kennzahlen übersichtlich in drei nebeneinander angeordneten Boxen dar.
    # 5. Zeig mir bitte den Delta-Wert im Vergleich zu einer Referenz-Zufriedenheitsrate von 43.45, am besten mit einer Farbcodierung.

    st.markdown("---    ")
    st.subheader("Key Insights")
    
    # Calculate
    total_passengers = len(filtered_data)
    satisfied_count = filtered_data[filtered_data['satisfaction'] == 'satisfied'].shape[0]
    satisfaction_rate = (satisfied_count / total_passengers) * 100
    service_means_insights = filtered_data.groupby("satisfaction")[service_cols].mean()
    
    # Find the service with biggest difference between satisfied and dissatisfied
    service_gap = service_means_insights.loc['satisfied'] - service_means_insights.loc['neutral or dissatisfied']
    biggest_gap_service = service_gap.idxmax()
    biggest_gap_value = service_gap.max()
    
    # Find average delay time difference
    avg_delay_satisfied = filtered_data[filtered_data['satisfaction'] == 'satisfied']['Departure Delay in Minutes'].mean()
    avg_delay_dissatisfied = filtered_data[filtered_data['satisfaction'] == 'neutral or dissatisfied']['Departure Delay in Minutes'].mean()
    delay_difference = avg_delay_dissatisfied - avg_delay_satisfied
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Overall Satisfaction Rate", 
            f"{satisfaction_rate:.1f}%", 
            delta=f"{satisfaction_rate - 43.45:.1f}%" if satisfaction_rate != 43.45 else "0%")
    
    with col2:
        st.metric(
            "Biggest Service Gap", 
            biggest_gap_service,
            f"+{biggest_gap_value:.1f} points")
    
    with col3:
        st.metric(
            "Delay Impact", 
            f"{delay_difference:.1f} min",
            "longer for dissatisfied passengers")
            
    st.markdown("<br>", unsafe_allow_html=True)


    # Add explanation of key findings
    # ----------------------------------------
    st.subheader("Analysis Summary")
    
    st.markdown(f"""
    
    Based on your selected filters ({len(filtered_data):,} passengers):
    
    - **{satisfaction_rate:.1f}%** of passengers reported being satisfied
    - The biggest difference in service ratings is for **{biggest_gap_service}** with satisfied passengers rating it **{biggest_gap_value:.1f} points higher**
    - Dissatisfied passengers experienced on average **{delay_difference:.1f} minutes more delay** than satisfied ones
    """)
    

    # Export Filtered Data
    # ----------------------------------------
    st.markdown("---    ")
    st.subheader("Export Filtered Data")
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode("utf-8")

    csv = convert_df_to_csv(filtered_data)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name="filtered_airline_satisfaction_data.csv",
        mime="text/csv",)


# --------- Satisfaction Prediction ---------
# -------------------------------------------

# Prompt an ChatGPT
# „Erstelle in Streamlit eine Benutzeroberfläche, in der Nutzer demografische Informationen und Flugdaten über die Sidebar eingeben können. 
# Verwende Slider und Selectboxen für Alter, Geschlecht, Kundentyp, Flugdistanz, Reiseklasse, Reisetyp sowie ein Number Input für Verspätung. 
# Initialisiere vorher ein Modell (simple_model), einen Scaler und eine Liste mit Modell-Features.“

elif page == "Satisfaction Prediction":
    st.title("Passenger Satisfaction Prediction")
    st.image("prediction.jpg", use_column_width=True)  
    st.markdown("""
    Use this interactive tool to **predict individual passenger satisfaction** based on demographic and service experience inputs.  
    You can simulate realistic customer scenarios to:

    - Evaluate the **probability of dissatisfaction** based on current service ratings  
    - Identify the **most influential service drivers** for each case  
    - Receive **targeted improvement suggestions** based on low-scoring and important features  
    - Review detailed customer profile information and export scenarios for further training or quality initiatives  

    This feature enables proactive decision-making by simulating outcomes before they occur – perfect for customer service training and experience management.
    """)
    
    simple_model, scaler, model_features = create_simple_model() #Create a simple model on the fly instead of using the XGBoost model
    

    # Move Customer Demographics to Sidebar
    # ----------------------------------------
    st.sidebar.header("Customer Profile")
    age = st.sidebar.slider("Age", 18, 100, 35)
    gender = st.sidebar.selectbox("Gender", data["Gender"].unique())
    customer_type = st.sidebar.selectbox("Customer Type", data["Customer Type"].unique())
    flight_distance = st.sidebar.slider("Flight Distance (km)", 0, 5000, 1500)
    class_type = st.sidebar.selectbox("Travel Class", data["Class"].unique())
    travel_type = st.sidebar.selectbox("Type of Travel", data["Type of Travel"].unique())
    
    st.sidebar.markdown("---")
    st.sidebar.header("Delay Information")
    dep_delay = st.sidebar.number_input("Departure Delay (minutes)", 0, 500, 0)
    

    # Main Layout for Service Ratings
    # ----------------------------------------
    st.markdown("---    ")
    st.subheader("Service Experience Ratings (1-5)")
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### Booking & Boarding")
        booking = st.slider("Ease of Online Booking", 1, 5, 3, key="booking")
        boarding = st.slider("Online Boarding", 1, 5, 3, key="boarding")
        checkin = st.slider("Check-in Service", 1, 5, 3, key="checkin")
        gate = st.slider("Gate Location", 1, 5, 3, key="gate")
        baggage = st.slider("Baggage Handling", 1, 5, 3, key="baggage")
    
    with col2:
        st.markdown("##### In-flight Comfort")
        comfort = st.slider("Seat Comfort", 1, 5, 3, key="comfort")
        legroom = st.slider("Leg Room Service", 1, 5, 3, key="legroom")
        cleanliness = st.slider("Cleanliness", 1, 5, 3, key="cleanliness")
        wifi = st.slider("Inflight WiFi Service", 1, 5, 3, key="wifi")
    
    with col3:
        st.markdown("##### Service & Amenities")
        food = st.slider("Food and Drink", 1, 5, 3, key="food")
        entertainment = st.slider("Inflight Entertainment", 1, 5, 3, key="entertainment")
        service = st.slider("On-board Service", 1, 5, 3, key="service")
        inflight_service = st.slider("Inflight Service", 1, 5, 3, key="inflight_service")
        timing = st.slider("Departure/Arrival Time Convenient", 1, 5, 3, key="timing")
    
    st.markdown("---    ")
    st.markdown("<br>", unsafe_allow_html=True)
    

    # Prediction Button
    # ----------------------------------------

    # ChatGPT Prompt:
    # Füge eine Schaltfläche hinzu, mit der eine Zufriedenheitsvorhersage ausgelöst wird. 
    # Beim Klick sollen die Nutzereingaben in ein DataFrame umgewandelt, skaliert und mit einem Klassifikationsmodell vorhergesagt werden. 
    # Zeige das Ergebnis (Zufrieden/Unzufrieden) inklusive Konfidenz, eine kurze Profilzusammenfassung sowie Verbesserungsvorschläge für schlecht bewertete, aber wichtige Servicefaktoren. 
    # Visualisiere zusätzlich die global wichtigsten Einflussfaktoren als horizontales Balkendiagramm mit Plotly.“

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_button = st.button("Predict Passenger Satisfaction", type="primary", use_container_width=True)
    
    # Prediction results section with improved styling
    if predict_button:
        try:
            # Create input data with the features the model expects
            input_data = {
                'Age': age,
                'Flight Distance': flight_distance,
                'Inflight wifi service': wifi,
                'Departure/Arrival time convenient': timing,
                'Ease of Online booking': booking,
                'Gate location': gate,
                'Food and drink': food,
                'Online boarding': boarding,
                'Seat comfort': comfort,
                'Inflight entertainment': entertainment,
                'On-board service': service,
                'Leg room service': legroom,
                'Baggage handling': baggage,
                'Checkin service': checkin,
                'Inflight service': inflight_service,
                'Cleanliness': cleanliness,
                'Departure Delay in Minutes': dep_delay
            }
            
            # Create DataFrame with the input data
            input_df = pd.DataFrame([input_data])
            
            # Scale the input data
            input_scaled = scaler.transform(input_df)
            
            # Make prediction with the simple model
            prediction = simple_model.predict(input_scaled)
            prediction_proba = simple_model.predict_proba(input_scaled)
            
            # Display prediction results in a nice box
            st.markdown("---    ")
            st.subheader("Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction[0] == 1:
                    st.success(f"### Customer is likely to be SATISFIED! 😊\n\n**Confidence: {prediction_proba[0][1]*100:.1f}%**")
                else:
                    st.error(f"### Customer is likely to be DISSATISFIED! 😐\n\n**Confidence: {prediction_proba[0][0]*100:.1f}%**")
                
                # Display customer profile summary
                st.markdown(f"""
                **Customer Profile:**
                - {age} years old, {gender}
                - {customer_type}
                - {travel_type}, {class_type} Class
                - Flight Distance: {flight_distance} km
                - Departure Delay: {dep_delay} min
                """)
            
            with col2:
                # Get feature importances
                importances = simple_model.feature_importances_
                feature_importance = pd.Series(importances, index=model_features)
                top_features = feature_importance.sort_values(ascending=False)[:5]
                
                # Show service improvement suggestions
                st.subheader("Service Improvement Suggestions")
                
                # Simplified analysis based on ratings
                low_ratings = []
                for feature, importance in top_features.items():
                    if feature in input_data and input_data[feature] < 3:
                        low_ratings.append((feature, input_data[feature], importance))
                
                if low_ratings:
                    st.markdown("**Areas needing improvement (based on importance):**")
                    for feature, rating, importance in sorted(low_ratings, key=lambda x: x[2], reverse=True):
                        st.markdown(f"- **{feature}**: Currently rated {rating}/5")
                else:
                    st.markdown("✅ **All important service ratings are above average!**")
                    
                    # Find any ratings below 3 even if not in top features
                    any_low_ratings = [(f, v) for f, v in input_data.items() 
                                      if f in model_features and v < 3]
                    if any_low_ratings:
                        st.markdown("**Other areas that could use improvement:**")
                        for feature, rating in any_low_ratings:
                            st.markdown(f"- **{feature}**: Currently rated {rating}/5")

            st.markdown("<br>", unsafe_allow_html=True)
            
            # Show top features visualization
            st.subheader("Globally Most Important Factors")
            
            top_10_features = feature_importance.sort_values(ascending=False).head(10).sort_values()
            
            fig = go.Figure(data=[
                go.Bar(
                    x=top_10_features.values,
                    y=top_10_features.index,
                    orientation='h',
                    marker_color="#003566",
                    text=[f"{val:.2f}" for val in top_10_features.values],
                    textposition="auto")])

            fig.update_layout(
                title="Top 10 Factors Influencing Satisfaction",
                title_x=0.4, 
                yaxis_title="Feature",
                height=450,
                margin=dict(l=40, r=40, t=60, b=40),
                font=dict(color="#003566"))

            st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.write("Please ensure all fields are filled correctly.")
            import traceback
            st.error(f"Details: {traceback.format_exc()}")


# --------- Customer Journey Map ------------
# -------------------------------------------

# ChatGPT Prompt:
# Erstelle eine visuelle Customer Journey Map in Form eines Scatterplots mit Plotly, basierend auf durchschnittlichen Zufriedenheitswerten und deren Einfluss auf die Gesamtzufriedenheit. 
# Gliedere die Touchpoints in drei Phasen (Pre-Flight, In-Flight, Post-Flight), färbe sie entsprechend, skaliere die Punktgrößen nach Impact, und verbinde die Punkte mit einer Linie. 
# Markiere außerdem eine Zufriedenheitsschwelle bei y=3 und hinterlege die Phasenbereiche farbig mit Annotationen.

elif page == "Customer Journey Map":
    st.title("Customer Journey Map")
    st.image("journey.jpg", use_column_width=True)
    st.markdown("""
    This section visualizes the **end-to-end customer journey** across key service phases: **Pre-Flight**, **In-Flight**, and **Post-Flight**. Using average satisfaction scores, you can:

    - View how passengers experience each **touchpoint** (e.g. check-in, seat comfort, baggage)
    - Identify **pain points** that fall below the satisfaction threshold
    - Understand the **relative importance** of each touchpoint in shaping the overall experience
    - Highlight **critical focus areas** (low-performing and high-impact) for improvement
    - Extract quick insights on what works well and what needs attention

    This view supports operational teams and service designers in optimizing each phase of the passenger journey.
    """)
    
    # Customer Journey Map
    # ----------------------------------------
    st.markdown("---    ")
    st.subheader("Customer Journey Overview")
    
    # Touchpoints & metadata
    journey_phases = {
        "Pre-Flight": ["Online Booking", "Online Boarding", "Check-in", "Baggage Drop", "Security", "Gate Location"],
        "In-Flight": ["Seat Comfort", "WiFi", "Food & Drink", "Entertainment", "Cabin Service"],
        "Post-Flight": ["Arrival", "Baggage Claim", "Exit"]}

    # Satisfaction scores for each touchpoint (from the dataset)
    touchpoint_scores = {
        "Online Booking": data["Ease of Online booking"].mean(),
        "Check-in": data["Checkin service"].mean(),
        "Baggage Drop": data["Baggage handling"].mean() * 0.9,  # Slightly lower than baggage handling overall
        "Security": 3.2,  # Simulated score
        "Gate Location": data["Gate location"].mean(),
        "Online Boarding": data["Online boarding"].mean(),
        "Seat Comfort": data["Seat comfort"].mean(),
        "WiFi": data["Inflight wifi service"].mean(),
        "Food & Drink": data["Food and drink"].mean(),
        "Entertainment": data["Inflight entertainment"].mean(),
        "Cabin Service": (data["On-board service"].mean() + data["Inflight service"].mean()) / 2,
        "Arrival": data["Departure/Arrival time convenient"].mean(),
        "Baggage Claim": data["Baggage handling"].mean(),
        "Exit": 3.8}  # Simulated score
    
    # Impact on overall satisfaction
    touchpoint_impact = {
        "Online Booking": 0.17,
        "Check-in": 0.24,
        "Baggage Drop": 0.25,
        "Security": 0.24,
        "Gate Location": 0.01,
        "Online Boarding": 0.50,
        "Seat Comfort": 0.35,
        "WiFi": 0.28,
        "Food & Drink": 0.21,
        "Entertainment": 0.40,
        "Cabin Service": 0.32,
        "Arrival": 0.05,
        "Baggage Claim": 0.25,
        "Exit": 0.25}

    # Build dataframe
    data = []
    for phase, touchpoints in journey_phases.items():
        for tp in touchpoints:
            data.append({
                "Phase": phase,
                "Touchpoint": tp,
                "Score": touchpoint_scores[tp],
                "Impact": touchpoint_impact[tp],
            })
    df = pd.DataFrame(data)
    df["Order"] = range(len(df))
    df["Size"] = df["Impact"] * 1800

    colors = {"Pre-Flight": "#1976D2", "In-Flight": "#388E3C", "Post-Flight": "#E64A19"}

    # Plotly Chart
    fig = px.scatter(
        df,
        x="Order",
        y="Score",
        size="Size",
        color="Phase",
        color_discrete_map=colors,
        text="Touchpoint",
        hover_name="Touchpoint",
        size_max=60)

    # Add vertical background rectangles for each phase
    for phase, group in df.groupby("Phase"):
        fig.add_vrect(
            x0=group["Order"].min() - 0.5,
            x1=group["Order"].max() + 0.5,
            fillcolor=colors[phase],
            opacity=0.06,
            layer="below",
            line_width=0,
            annotation_text=phase,
            annotation_position="top left",
            annotation_font=dict(size=14, color=colors[phase]))

    # Add satisfaction threshold line
    fig.add_shape(
        type="line",
        x0=-0.5,
        x1=len(df)-0.5,
        y0=3,
        y1=3,
        line=dict(color="red", dash="dash"))

    fig.add_annotation(
        x=len(df)-1,
        y=3.05,
        text="Satisfaction Threshold",
        showarrow=False,
        font=dict(size=11, color="red"),
        xanchor="right")

    # Draw the journey line connecting all touchpoints
    x_smooth = np.linspace(df["Order"].min(), df["Order"].max(), 100)
    y_smooth = np.interp(x_smooth, df["Order"], df["Score"])
    fig.add_trace(
        go.Scatter(
            x=x_smooth,
            y=y_smooth,
            mode="lines",
            line=dict(color="gray", width=2),
            name="Journey Flow",
            showlegend=False))

    # Final layout styling
    fig.update_traces(
        textfont=dict(size=11, color="#003566"),
        textposition="top center",
        marker=dict(line=dict(width=1, color="white"))) 

    fig.update_layout(
        height=500,
        title="Customer Journey Satisfaction Map",
        title_x=0.4,
        xaxis=dict(title="", showticklabels=False),
        yaxis=dict(
            title="Customer Satisfaction Rating (1-5)",
            range=[1, 5],
            tickvals=[1, 2, 3, 4, 5],
            ticktext=["1 (Poor)", "2", "3 (Average)", "4", "5 (Excellent)"]
        ),
        margin=dict(l=40, r=40, t=60, b=40),
        legend_title="Journey Phase")

    st.plotly_chart(fig, use_container_width=True)

    
    # How to Read This Map
    # ----------------------------------------

    # ChatGPT Prompt:
    # Füge unterhalb einer Customer Journey Map eine erklärende Legende hinzu, die beschreibt, wie man die Visualisierung liest (z. B. Bedeutung der Farben, Punktgrößen und Positionen). 
    # Erstelle anschließend eine Zwei-Spalten-Übersicht mit vier Analyseblöcken: 
    # ‘Pain Points’ (niedrigste Scores), ‘Most Impactful Touchpoints’ (höchster Einfluss), ‘Strong Areas’ (beste Scores) und ‘Critical Focus Areas’ (niedriger Score & hoher Impact). 
    # Verwende gestylte Markdown-Ausgaben zur besseren visuellen Hervorhebung

    st.markdown("""
    #### How to Read This Map
    
    - **Bubble Size**: Represents the impact each touchpoint has on overall satisfaction
    - **Position on Journey Line**: Shows chronological order of touchpoints
    - **Color**: Indicates journey phase (Pre-Flight, In-Flight, Post-Flight)
    - **Vertical Position**: Represents average satisfaction rating for each touchpoint
    - **Red Dashed Line**: Satisfaction threshold - touchpoints below this line require attention
    """)
    

    # Key Insights
    # ----------------------------------------
    st.markdown("---    ")
    st.subheader("Key Insights")
    
    # Touchpoint DataFrames
    touchpoint_scores_df = pd.DataFrame(list(touchpoint_scores.items()), columns=['Touchpoint', 'Score'])
    lowest_touchpoints = touchpoint_scores_df.sort_values('Score').head(3)
    highest_touchpoints = touchpoint_scores_df.sort_values('Score', ascending=False).head(3)

    touchpoint_impact_df = pd.DataFrame(list(touchpoint_impact.items()), columns=['Touchpoint', 'Impact'])
    highest_impact = touchpoint_impact_df.sort_values('Impact', ascending=False).head(3)

    # Critical Focus Areas (Low Score + High Impact)
    critical_touchpoints = []
    for tp in touchpoint_scores:
        if touchpoint_scores[tp] < 3.3 and touchpoint_impact[tp] > 0.08:
            critical_touchpoints.append((tp, touchpoint_scores[tp], touchpoint_impact[tp]))

    # Display in 2-column layout with stylized markdown
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### ❗ Pain Points")
        for _, row in lowest_touchpoints.iterrows():
            st.markdown(f"<span style='font-weight:bold; color:#003566;'>{row['Touchpoint']}</span> <span style='color:gray;'>({row['Score']:.2f}/5)</span>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("#### 📊 Most Impactful Touchpoints")
        for _, row in highest_impact.iterrows():
            st.markdown(f"<span style='font-weight:bold; color:#003566;'>{row['Touchpoint']}</span> <span style='color:gray;'>({row['Impact']:.2f} impact)</span>", unsafe_allow_html=True)

    with col2:
        st.markdown("#### ✅ Strong Areas")
        for _, row in highest_touchpoints.iterrows():
            st.markdown(f"<span style='font-weight:bold; color:#003566;'>{row['Touchpoint']}</span> <span style='color:gray;'>({row['Score']:.2f}/5)</span>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("#### 🔧 Critical Focus Areas")
        for tp, score, impact in sorted(critical_touchpoints, key=lambda x: x[2], reverse=True):
            st.markdown(f"<span style='font-weight:bold; color:#003566;'>{tp}</span> <span style='color:gray;'>(Score: {score:.2f}, Impact: {impact:.2f})</span>", unsafe_allow_html=True)


# ------------ Recommendations --------------
# -------------------------------------------

# ChatGPT Prompt:
# „Erstelle eine Empfehlungsseite mit datenbasierten strategischen Handlungsvorschlägen. 
# Berechne dafür die Korrelation numerischer Variablen mit der Zufriedenheit und leite daraus die wichtigsten Einflussfaktoren ab. 
# Präsentiere konkrete Maßnahmen gegliedert in 'Quick Wins (0–3 Monate)' und 'Mid-Term Initiatives (3–12 Monate)' in formatierten Boxen. 
# Ergänze eine Gantt-Chart-ähnliche Roadmap zur Visualisierung der Umsetzung über 20 Monate hinweg, farbcodiert nach Priorität (hoch/mittel/niedrig).“

elif page == "Recommendations":
    st.title("Strategic Recommendations")
    st.image("recommendations.jpg", use_column_width=True)
    st.markdown("Data-driven insights and actionable recommendations for management")
    
    # Define a function to analyze feature importance
    def analyze_feature_importance(data):
        # Calculate correlation with satisfaction
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != "satisfaction_numeric"]
        
        correlations = data[numeric_cols + ["satisfaction_numeric"]].corr()
        importance = correlations["satisfaction_numeric"].drop("satisfaction_numeric").sort_values(ascending=False)
        
        return importance
    
    feature_importance = analyze_feature_importance(data) # Get feature importance - using our new function
    
    # Top Priorities Areas
    # ----------------------------------------
    st.markdown("---    ")
    st.subheader("Priority Action Areas")
    
    high_impact_services = feature_importance.head(5) # High impact services - top 5 features by correlation with satisfaction
    
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h3>🎯 Quick Wins (0–3 months)</h3>
            <ul>
                <li><strong>WiFi Service:</strong> Improve connectivity</li>
                <li><strong>Cleanliness:</strong> Enhance hygiene standards</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px;">
            <h3>📈 Mid-Term Initiatives (3–12 months)</h3>
            <ul>
                <li><strong>Online Boarding Process:</strong> Streamline digital check-in</li>
                <li><strong>Seat Comfort:</strong> Upgrade seat cushions</li>
                <li><strong>Entertainment System:</strong> Refresh content & displays</li>
                <li><strong>Food Service:</strong> Redesign in-flight menu</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

 
    # Implementation Roadmap
    # ----------------------------------------
    st.markdown("---    ")
    st.subheader("Implementation Roadmap")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Create Gantt-style chart using matplotlib
    initiatives = ['WiFi Upgrade', 'Seat Replacement', 'Improve Hygiene', 'Digital Platform', 'Entertainment System', 'Revised Food Menu']
    start_months = [1, 6, 1, 1, 4, 3]
    durations = [3, 12, 3, 9, 8, 6]
    priorities = ['High', 'Medium', 'High', 'High', 'Low', 'Medium']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = {'High': '#f44336', 'Medium': '#ff9800', 'Low': '#4CAF50'}
    
    for i, (initiative, start, duration, priority) in enumerate(zip(initiatives, start_months, durations, priorities)):
        ax.barh(i, duration, left=start, height=0.5, 
                color=colors[priority], alpha=0.7, label=priority if i == 0 else "")
        ax.text(start + duration/2, i, initiative, ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Timeline (Months)', fontsize=12)
    ax.set_ylabel('Initiatives', fontsize=12)
    ax.set_title('Service Improvement Implementation Roadmap', fontsize=14)
    ax.set_yticks(range(len(initiatives)))
    ax.set_yticklabels(initiatives)
    ax.set_xlim(0, 20)
    ax.grid(axis='x', alpha=0.5)
    
    # Create custom legend
    handles = [plt.Rectangle((0,0),1,1, color=colors[label], alpha=0.7) for label in colors]
    ax.legend(handles, colors.keys(), title="Priority", loc="upper right")
    
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>Airline Passenger Satisfaction Analyse | Universität St. Gallen, Big Data & Data Science 
            </p>
    <p>Letzte Aktualisierung {}</p>
</div>
""".format(datetime.now().strftime("%Y-%m-%d")), unsafe_allow_html=True)
