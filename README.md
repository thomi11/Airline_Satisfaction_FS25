# Airline Passenger Satisfaction Analytics

## Übersicht
Diese interaktive Streamlit-Anwendung bietet umfassende Analysen und Prognosen zur Kundenzufriedenheit im Flugverkehr. Sie ermöglicht es Fluggesellschaften und Analysten, Kundenerfahrungen zu visualisieren, Einflussfaktoren zu identifizieren und Verbesserungsstrategien zu entwickeln.

## Features

### Executive Dashboard
- **KPI-Übersicht**: Zufriedenheitsrate, Net Satisfaction Score, durchschnittliche Verspätungen und Kundenbindungsraten auf einen Blick
- **Zufriedenheitsverteilung**: Visuelle Darstellung der Kundensentiments nach verschiedenen Segmenten
- **Feature-Importance-Analyse**: Identifikation der Top-10-Faktoren, die Kundenzufriedenheit beeinflussen
- **Datenexport**: Möglichkeit, gefilterte Daten als CSV-Datei zu exportieren

### Detailed Analysis
- **Service-Heatmap**: Vergleich der durchschnittlichen Servicebewertungen zwischen zufriedenen und unzufriedenen Passagieren
- **Verteilungsanalyse**: Detaillierte Auswertung einzelner Servicefaktoren nach Zufriedenheitskategorie
- **Verspätungsanalyse**: Untersuchung des Einflusses von Verspätungen auf die Kundenzufriedenheit
- **Insight-Zusammenfassung**: Automatische Berechnung und Darstellung der wichtigsten Erkenntnisse

### Satisfaction Prediction
- **Vorhersagemodell**: Machine-Learning-basierte Prognose der Kundenzufriedenheit
- **Interaktive Eingabe**: Simulation verschiedener Kundenszenarien und Servicebedingungen
- **Verbesserungsvorschläge**: Automatische Empfehlungen für Bereiche mit niedrigen Bewertungen
- **Feature-Importance-Visualisierung**: Darstellung der wichtigsten Einflussfaktoren auf die Zufriedenheit

### Customer Journey Map
- **End-to-End-Visualisierung**: Darstellung des gesamten Kundenerlebnisses über verschiedene Touchpoints
- **Phasenunterteilung**: Analyse der Pre-Flight-, In-Flight- und Post-Flight-Erfahrungen
- **Schwachstellenidentifikation**: Hervorhebung von Touchpoints unterhalb der Zufriedenheitsschwelle
- **Impact-Analyse**: Größenbasierte Darstellung des Einflusses jedes Touchpoints auf die Gesamtzufriedenheit

### Recommendations
- **Datenbasierte Strategien**: Aus Kundenanalysen abgeleitete Handlungsempfehlungen
- **Priorisierter Aktionsplan**: Unterteilung in kurzfristige "Quick Wins" und mittelfristige Initiativen
- **Implementierungs-Roadmap**: Visualisierte Zeitplanung für die Umsetzung von Verbesserungsmaßnahmen
- **Priorisierungsrahmen**: Farbkodierte Darstellung der Maßnahmen nach Wichtigkeit

## Installation

```bash
# Repository klonen
git clone https://github.com/[IHR_USERNAME]/airline-satisfaction-analytics.git
cd airline-satisfaction-analytics

# Virtuelle Umgebung erstellen und aktivieren (optional, aber empfohlen)
python -m venv venv
source venv/bin/activate  # Unter Windows: venv\Scripts\activate

# Abhängigkeiten installieren
pip install -r requirements.txt
```

## Ausführung

```bash
streamlit run capstone.py
```

## Datenstruktur

Die Anwendung verwendet den Datensatz `capstone_merged_data.csv` mit folgenden Hauptmerkmalen:

- **Demografische Daten**: Geschlecht, Alter, Kundentyp
- **Fluginformationen**: Fluglänge, Reiseklasse, Reisetyp, Verspätungen
- **Servicebewertungen**: 14 verschiedene Servicebereiche auf einer Skala von 1-5
- **Zielvariable**: Zufriedenheitsstatus (zufrieden / neutral oder unzufrieden)

## Modell

Die App verwendet zwei Modelle:
1. Ein XGBoost-Modell (`xgb_model.sav`), das für eine genaue Vorhersage der Kundenzufriedenheit trainiert wurde
2. Ein einfacheres Random-Forest-Modell, das zur Echtzeitanalyse von Benutzereingaben erstellt wird

## Anpassung und Weiterentwicklung

Die Anwendung kann leicht angepasst werden, um spezifische Anforderungen zu erfüllen:

- Ändern der Farbschemata durch Anpassung der CSS-Definitionen
- Hinzufügen neuer Analysemetriken in bestehende Module
- Integration zusätzlicher Datenquellen durch Modifikation der Datenlademodule
- Erweiterung um neue Visualisierungen und Dashboards

## Technologien

- **Streamlit**: Web-Framework für Datenapplikationen
- **Pandas & NumPy**: Datenverarbeitung und -analyse
- **Plotly & Matplotlib**: Interaktive Visualisierungen
- **Scikit-learn & XGBoost**: Machine Learning und Vorhersagen

## Mitwirkende

Diese Anwendung wurde als Capstone-Projekt für den Kurs "Big Data & Data Science" an der Universität St. Gallen entwickelt.

***
# Hilfmittel:
Chatgpt

# ChatGPT Prompt:
Basierend auf meinem .py file, erstelle mir eine readme description für mein Airline Projekt
