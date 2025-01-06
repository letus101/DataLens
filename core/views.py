import os
from django.conf import settings
from django.shortcuts import render, redirect, HttpResponse
import pandas as pd  # Note the corrected import statement
import requests
from .forms import BinomialForm, FileUploadForm, ExponentielleForm, TraitementForm, UniformeForm, PoissonForm, \
    NormaleForm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO, BytesIO
import base64
import json
import plotly.express as px
import matplotlib
import plotly.graph_objs as go
from scipy.stats import binom
from django.http import JsonResponse
import plotly.io as pio
from .forms import BernoulliForm
from scipy.stats import bernoulli, norm, t, expon, poisson, uniform
from scipy import stats
from unittest import result

matplotlib.use('Agg')
from statistics import mean, median, mode, variance, stdev
import plotly.figure_factory as ff
import io


def index(request):
    return render(request, 'index.html')


def generate_chart(df, type_chart, col1, col2):
    buffer = BytesIO()

    if type_chart == 'Barplot':
        fig = px.bar(df, x=col1, y=col2)
        fig.update_layout(xaxis_title=col1, yaxis_title=col2, title='Bar Plot')
        return fig.to_json()

    elif type_chart == 'histogram':
        fig = px.histogram(df, x=col1)
        fig.update_layout(xaxis_title=col1, yaxis_title='Count', title='Histogram', barmode='overlay', bargap=0.1)
        return fig.to_json()

    elif type_chart == 'piechart':
        value_counts = df[col1].value_counts().reset_index()
        value_counts.columns = [col1, 'Count']
        fig = px.pie(value_counts, values='Count', names=col1, title='Pie Chart')
        return fig.to_json()


    elif type_chart == 'scatterplot':
        fig = px.scatter(df, x=col1, y=col2)
        fig.update_layout(xaxis_title=col1, yaxis_title=col2, title='Scatter Plot')
        return fig.to_json()

    elif type_chart == 'heatmap':
        df_encoded = df.copy()
        for column in df_encoded.columns:
            if df_encoded[column].dtype == 'object':
                df_encoded[column], _ = pd.factorize(df_encoded[column])
        fig = px.imshow(df_encoded.corr(), color_continuous_scale='Viridis')
        fig.update_layout(title='Heatmap')
        return fig.to_json()

    elif type_chart == 'lineplot':

        fig = px.line(df, x=col1, y=col2, markers=True)
        fig.update_layout(xaxis_title=col1, yaxis_title=col2, title='Line Plot')
        return fig.to_json()


    elif type_chart == 'boxplot':
        fig = px.box(df, x=col1)
        fig.update_layout(title='Box Plot')
        return fig.to_json()


    elif type_chart == 'violinplot':
        fig = px.violin(df, y=col1, box=True)
        fig.update_layout(yaxis_title=col1, title='Violin Plot')
        return fig.to_json()




    elif type_chart == 'kdeplot':
        data_to_plot = df[col1].replace([np.inf, -np.inf], np.nan).dropna()

        group_labels = ['distplot']  # This will be the label for the distribution
        # Generate the KDE plot with the histogram
        fig = ff.create_distplot([data_to_plot], group_labels, curve_type='kde', show_hist=True,
                                 histnorm='probability density')

        # Mise à jour de la disposition (layout) pour correspondre au style souhaité
        fig.update_layout(
            title="Kernel Density Estimation (KDE) Plot",
            yaxis_title="Density",
            xaxis_title=col1,
            showlegend=False,
            template='plotly_white'
        )

        # Update traces to match the desired style
        fig.update_traces(marker=dict(color='grey', line=dict(color='black', width=1.5)))

        fig_json = fig.to_json()

        return fig_json


def excel(request):
    if request.method == 'POST':
        form = FileUploadForm(request.POST, request.FILES)

        if form.is_valid():
            fichier = request.FILES['file']

            if fichier.name.endswith(('.xls', '.xlsx')):
                try:
                    data = pd.read_excel(fichier)
                    df = pd.DataFrame(data)
                    columns_choices = [(col, col) for col in df.columns]
                    df_json = df.to_json()
                    request.session['df_json'] = df_json
                    return render(
                        request,
                        'visualiser_data.html',
                        {'form': form, 'df': df.to_html(classes='table table-bordered'), 'column_names': df.columns},
                    )
                except pd.errors.ParserError as e:
                    e = f"Erreur : Impossible de lire le fichier Excel. Assurez-vous que le fichier est au format Excel valide."
                    return render(request, 'excel.html', {'form': form, 'error_message': e})
            else:
                return HttpResponse(
                    "Seuls les fichiers Excel (.xls, .xlsx) sont autorisés. Veuillez télécharger un fichier Excel.")
    else:
        form = FileUploadForm()

    return render(request, 'excel.html', {'form': form})


def visualiser(request):
    return render(request, 'visualiser_data.html')


def visualiser_chart(request):
    if request.method == 'POST':
        col1 = request.POST['col_name1']
        col2 = request.POST['col_name2']
        type_chart = request.POST['type_chart']
        df_json = request.session.get('df_json')

        df_json_io = StringIO(df_json)
        df = pd.read_json(df_json_io)

        if pd.api.types.is_string_dtype(df[col1]) and type_chart in ['kdeplot', 'violinplot', 'boxplot']:
            error_message = f"La colonne choisie est '{col1}'de type 'string', veuillez choisir une autre colonne."
            return render(request, 'diagramme.html', {'error_message': error_message})
        elif pd.api.types.is_string_dtype(df[col2]) and type_chart in ['Barplot', 'lineplot']:
            error_message = f"La deuxième colonne '{col2}' doit contenir des valeurs numériques. Veuillez choisir une autre colonne."
            return render(request, 'diagramme.html', {'error_message': error_message})
        elif type_chart == 'scatterplot':
            # Vérifier si l'une des colonnes n'est pas numérique
            if pd.api.types.is_string_dtype(df[col1]) or pd.api.types.is_string_dtype(df[col2]):
                # Préparer un message d'erreur en indiquant les noms des colonnes non numériques
                non_numeric_columns = [col for col in [col1, col2] if pd.api.types.is_string_dtype(df[col])]
                error_message = f"Les colonnes '{', '.join(non_numeric_columns)}' doivent être numériques. Veuillez choisir d'autres colonnes."
                return render(request, 'diagramme.html', {'error_message': error_message})

        elif type_chart == "Nothing":
            error_message = "Veuillez sélectionner un diagramme à afficher"
            return render(request, 'diagramme.html', {'error_message': error_message})

        chart = generate_chart(df, type_chart, col1, col2)
        return render(request, 'diagramme.html', {'chart': chart})

    return render(request, 'visualiser_data.html')


def diagramme(request):
    return render(request, 'diagramme.html')


def text(request):
    if request.method == 'POST':
        form = FileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            fichier = request.FILES['file']

            if fichier.name.endswith('.txt'):

                data = pd.read_csv(fichier)

                df = pd.DataFrame(data)
                columns_choices = [(col, col) for col in df.columns]
                df_json = df.to_json()
                request.session['df_json'] = df_json
                return render(
                    request,
                    'visualiser_data.html',
                    {'form': form, 'df': df.to_html(classes='table table-bordered'), 'column_names': df.columns},
                )
            else:
                return HttpResponse("Seuls les fichiers text sont autorisés. Veuillez télécharger un fichier txt.")
    else:
        form = FileUploadForm()

    return render(request, 'text.html', {'form': form})


def csv(request):
    if request.method == 'POST':
        form = FileUploadForm(request.POST, request.FILES)
        if form.is_valid():
            fichier = request.FILES['file']
            if fichier.name.endswith('.csv'):
                # Traitez le fichier CSV
                data = pd.read_csv(fichier)
                df = pd.DataFrame(data)
                columns_choices = [(col, col) for col in df.columns]
                df_json = df.to_json()
                request.session['df_json'] = df_json

                return render(
                    request,
                    'visualiser_data.html',
                    {'form': form, 'df': df.to_html(classes='table table-bordered'), 'column_names': df.columns},
                )
            else:
                return HttpResponse("Seuls les fichiers CSV sont autorisés. Veuillez télécharger un fichier CSV.")
    else:
        form = FileUploadForm()

    return render(request, 'csv.html', {'form': form})


def parcourir_chart(request):
    df = None
    columns_choices = None
    error_message = ""
    max_row = 0

    if 'df_json' in request.session:
        df_json = request.session['df_json']
        df = pd.read_json(StringIO(df_json))
        columns_choices = [col for col in df.columns]
        max_row = df.shape[0] - 1

    if request.method == 'POST':
        selected_columns = request.POST.getlist('selected_columns')
        parcourir_chart_type = request.POST.get('parcourir_chart')
        col_name1 = request.POST.get('col_name1')
        row_numb = request.POST.get('RowNumb')

        if selected_columns:
            df = df[selected_columns]

        if parcourir_chart_type == 'GroupBy':
            numeric_column = request.POST.get('numeric_column')
            condition = request.POST.get('condition')
            value = request.POST.get('value')

            if numeric_column and condition and value:
                try:
                    grouped_df = df.groupby(numeric_column)
                    value = float(value)
                    if condition == '>':
                        df = grouped_df.filter(lambda x: x[numeric_column].mean() > value)
                    elif condition == '<':
                        df = grouped_df.filter(lambda x: x[numeric_column].mean() < value)
                    elif condition == '=':
                        df = grouped_df.filter(lambda x: x[numeric_column].mean() == value)
                except Exception as e:
                    error_message = f"Une erreur est survenue : {str(e)}"

            contexte = {
                'df': df.to_html(classes='table table-bordered') if df is not None else None,
                'column_names': columns_choices,
                'max_row': max_row,
                'error_message': error_message
            }
            return render(request, 'parcourir.html', contexte)

        elif parcourir_chart_type == 'GroupByMean':
            numeric_column = request.POST.get('numeric_column')
            if numeric_column:
                try:
                    df = df.groupby(numeric_column).mean().reset_index()
                except Exception as e:
                    error_message = f"Une erreur est survenue : {str(e)}"

            contexte = {
                'df': df.to_html(classes='table table-bordered') if df is not None else None,
                'column_names': columns_choices,
                'max_row': max_row,
                'error_message': error_message
            }
            return render(request, 'parcourir.html', contexte)

        if parcourir_chart_type == 'FindElem' and df is not None:
            try:
                row_numb = int(row_numb)
                row_numb = min(row_numb, max_row)
                resultats_recherche = df.at[row_numb, col_name1]
                contexte = {'resultat': resultats_recherche, 'column_names': columns_choices,
                            'df': df.to_html(classes='table table-bordered'), 'max_row': max_row}
                return render(request, 'parcourir.html', contexte)
            except (ValueError, KeyError, IndexError):
                pass

        parcourir_rows_type = request.POST.get('parcourir_rows')

        if parcourir_rows_type == 'NbrOfRowsTop':
            nb_rows_top = int(request.POST.get('Head'))
            df = df.head(nb_rows_top)
        elif parcourir_rows_type == 'NbrOfRowsBottom':
            nb_rows_bottom = int(request.POST.get('Tail'))
            df = df.tail(nb_rows_bottom)
        elif parcourir_rows_type == 'FromRowToRow':
            from_row = int(request.POST.get('FromRowNumb'))
            to_row = int(request.POST.get('ToRowNumb'))
            df = df.loc[from_row:to_row]

    contexte = {
        'df': df.to_html(classes='table table-bordered') if df is not None else None,
        'column_names': columns_choices,
        'max_row': max_row
    }
    return render(request, 'parcourir.html', contexte)

# //////////////////////////////// LOIS ////////////////////////////////////////////////////////////////

def Binomiale(request):
    plot_data = None  # Variable pour stocker les données du graphique encodées en base64
    if request.method == 'POST':
        form = BinomialForm(request.POST)
        if form.is_valid():
            n = form.cleaned_data['n']
            p = form.cleaned_data['p']

            # Générer des données de la distribution binomiale
            data_binom = binom.rvs(n=n, p=p, loc=0, size=1000)

            # Créer l'histogramme avec Seaborn
            sns.set(style="whitegrid")
            plt.figure(figsize=(6, 4))
            ax = sns.histplot(data_binom, kde=True, stat='probability')
            ax.set(xlabel='Binomial', ylabel='Probabilité')

            # Sauvegarder la figure dans un buffer temporaire
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()

            # Encodage en base64 et décodage en UTF-8 pour la rendre utilisable dans le HTML
            plot_data = base64.b64encode(image_png).decode('utf-8')

    else:
        form = BinomialForm()

    return render(request, 'binomiale.html', {'form': form, 'plot_data': plot_data})


def Bernoulli(request):
    plot_data = None  # Variable pour stocker les données du graphique encodées en base64
    if request.method == 'POST':
        form = BernoulliForm(request.POST)
        if form.is_valid():
            p = form.cleaned_data['p']

            # Générer des données de la distribution de Bernoulli
            data_bern = bernoulli.rvs(size=1000, p=p)

            # Créer l'histogramme avec Seaborn
            sns.set(style="whitegrid")
            plt.figure(figsize=(6, 4))
            ax = sns.histplot(data_bern, kde=True, stat='probability')
            ax.set(xlabel='Bernoulli', ylabel='Probabilité')

            # Sauvegarder la figure dans un buffer temporaire
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()

            # Encodage en base64 et décodage en UTF-8 pour la rendre utilisable dans le HTML
            plot_data = base64.b64encode(image_png).decode('utf-8')

    else:
        form = BernoulliForm()

    return render(request, 'bernoulli.html', {'form': form, 'plot_data': plot_data})


def Normale(request):
    plot_data = None  # Variable pour stocker les données du graphique encodées en base64
    if request.method == 'POST':
        form = NormaleForm(request.POST)
        if form.is_valid():
            mean = form.cleaned_data['mean']
            std_dev = form.cleaned_data['std_dev']

            # Points pour la courbe de la distribution normale
            x_values = np.linspace(mean - 3 * std_dev, mean + 3 * std_dev, 1000)
            y_values = norm.pdf(x_values, mean, std_dev)

            # Créer la courbe de la distribution normale remplie avec Matplotlib
            sns.set(style="whitegrid")
            plt.figure(figsize=(10, 6))
            plt.fill_between(x_values, y_values, color="skyblue", alpha=0.4)
            plt.plot(x_values, y_values, color="Slateblue", alpha=0.6)
            plt.title('Distribution Normale Continue')
            plt.xlabel('Valeur')
            plt.ylabel('Densité de probabilité')

            # Sauvegarder la figure dans un buffer temporaire
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()

            # Encodage en base64 et décodage en UTF-8 pour la rendre utilisable dans le HTML
            plot_data = base64.b64encode(image_png).decode('utf-8')

    else:
        form = NormaleForm()

    return render(request, 'normale.html', {'form': form, 'plot_data': plot_data})


def Poisson(request):
    plot_data = None  # Variable pour stocker les données du graphique encodées en base64
    if request.method == 'POST':
        form = PoissonForm(request.POST)
        if form.is_valid():
            lambda_param = form.cleaned_data['lambda_param']

            # Générer des données de la distribution de Poisson
            data_poisson = poisson.rvs(mu=lambda_param, size=1000)

            # Créer l'histogramme avec Seaborn
            sns.set(style="whitegrid")
            plt.figure(figsize=(6, 4))
            ax = sns.histplot(data_poisson, kde=True, stat='probability')
            ax.set(xlabel='Poisson', ylabel='Probabilité')

            # Sauvegarder la figure dans un buffer temporaire
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()

            # Encodage en base64 et décodage en UTF-8 pour la rendre utilisable dans le HTML
            plot_data = base64.b64encode(image_png).decode('utf-8')

    else:
        form = PoissonForm()

    return render(request, 'poisson.html', {'form': form, 'plot_data': plot_data})


def Uniforme(request):
    plot_data = None  # Variable pour stocker les données du graphique encodées en base64
    if request.method == 'POST':
        form = UniformeForm(request.POST)
        if form.is_valid():
            a = form.cleaned_data['a']
            b = form.cleaned_data['b']

            # Générer des données de la distribution uniforme
            data_unif = uniform.rvs(loc=a, scale=b - a, size=1000)  # b = loc + scale

            # Créer l'histogramme avec Seaborn
            sns.set(style="whitegrid")
            plt.figure(figsize=(6, 4))
            ax = sns.histplot(data_unif, kde=True, stat='probability')
            ax.set(xlabel='Uniforme', ylabel='Probabilité')

            # Sauvegarder la figure dans un buffer temporaire
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()

            # Encodage en base64 et décodage en UTF-8 pour la rendre utilisable dans le HTML
            plot_data = base64.b64encode(image_png).decode('utf-8')

    else:
        form = UniformeForm()

    return render(request, 'uniforme.html', {'form': form, 'plot_data': plot_data})


# Assurez-vous d'importer votre formulaire ici

def Exponentielle(request):
    plot_data = None  # Variable pour stocker les données du graphique encodées en base64
    if request.method == 'POST':
        form = ExponentielleForm(request.POST)
        if form.is_valid():
            beta = form.cleaned_data['beta']

            # Générer des échantillons de la distribution exponentielle
            data_exponentielle = expon.rvs(scale=beta, size=1000)

            # Créer la courbe de densité avec Seaborn
            sns.set(style="whitegrid")
            plt.figure(figsize=(6, 4))
            sns.kdeplot(data_exponentielle, fill=True)
            plt.title('Distribution Exponentielle')
            plt.xlabel('Valeur')
            plt.ylabel('Densité de probabilité')

            # Sauvegarder la figure dans un buffer temporaire
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            buffer.seek(0)
            image_png = buffer.getvalue()
            buffer.close()

            # Encodage en base64 et décodage en UTF-8 pour la rendre utilisable dans le HTML
            plot_data = base64.b64encode(image_png).decode('utf-8')

    else:
        form = ExponentielleForm()

    return render(request, 'exponentielle.html', {'form': form, 'plot_data': plot_data})


# /////////////////////////////////////////////////////////calcules

import numpy as np


def mode(valeurs):
    valeurs = np.array(valeurs.replace(';', ',').split(','), dtype=float)
    uniques, counts = np.unique(valeurs, return_counts=True)
    max_count = np.max(counts)
    modes = uniques[counts == max_count]
    if max_count == 1 or len(modes) == len(uniques):
        return "Il n'y a pas de mode"
    else:
        return modes.tolist()


def Calcules(request):
    if request.method == 'POST':
        form = TraitementForm(request.POST)
        if form.is_valid():
            valeurs_input = form.cleaned_data['valeurs']

            # Traiter les valeurs saisies
            valeurs = [float(x.strip()) for x in valeurs_input.replace(';', ',').split(',') if x.strip()]

            # Calcul des statistiques
            mean_value = np.mean(valeurs)
            median_value = np.median(valeurs)
            mode_value = mode(valeurs_input)
            variance_value = np.var(valeurs)
            stdev_value = np.std(valeurs)

            return render(request, 'calcules.html', {'form': form, 'mean': mean_value,
                                                     'median': median_value, 'mode': mode_value,
                                                     'variance': variance_value, 'stdev': stdev_value})
    else:
        form = TraitementForm()

    return render(request, 'calcules.html', {'form': form})


# /////////////////////////////////////////////testes
def calculate_z_test(field, sigma, n, significance):
    # Convertir les valeurs en nombres
    field = float(field)
    sigma = float(sigma)
    n = int(n)
    significance = float(significance)

    # Calculer le z-test
    z_stat = norm.sf(abs((field - 0) / (sigma / np.sqrt(n)))) * 2  # Two-tailed test

    # Interpréter les résultats du test
    if z_stat < significance:
        hypothesis_result = "Hypothesis rejected: The sample mean is significantly different from the population mean."
    else:
        hypothesis_result = "Hypothesis not rejected : There is no significant difference between the sample mean and the population mean."

    return z_stat, hypothesis_result


def calculate_z_test(field, zTestmi, sigma, n, significance):
    # Convertir les valeurs en nombres
    field = float(field)
    sigma = float(sigma)
    n = int(n)
    significance = float(significance)
    zTestmi = float(zTestmi.replace(',', '.'))
    # Calculer le z-test
    z_stat = (field - zTestmi) / (sigma / np.sqrt(n))

    # Calculer les p-values pour les trois cas
    p_value_two_sided = norm.sf(abs(z_stat)) * 2  # Bilatéral
    p_value_left = norm.cdf(z_stat)  # Unilatéral à gauche
    p_value_right = norm.sf(z_stat)  # Unilatéral à droite

    # Interpréter les résultats du test
    hypothesis_result_two_sided = "Hypothesis rejected: The sample mean is significantly different from the population mean." if p_value_two_sided < significance else "Hypothesis not rejected: There is no significant difference between the sample mean and the population mean."

    hypothesis_result_left = "Hypothesis rejected: The sample mean is significantly less than the population mean." if p_value_left < significance else "Hypothesis not rejected: There is no significant difference, or the sample mean is greater than the population mean."

    hypothesis_result_right = "Hypothesis rejected: The sample mean is significantly greater than the population mean." if p_value_right < significance else "Hypothesis not rejected: There is no significant difference, or the sample mean is less than the population mean."

    # Retourner les résultats du test sous forme de dictionnaire
    return {
        'z_statistic': z_stat,
        'p_value_two_sided': p_value_two_sided,
        'p_value_left': p_value_left,
        'p_value_right': p_value_right,
        'hypothesis_result_two_sided': hypothesis_result_two_sided,
        'hypothesis_result_left': hypothesis_result_left,
        'hypothesis_result_right': hypothesis_result_right
    }


def calculate_linear_regression(x_values, y_values):
    # Convert x_values and y_values to numpy arrays
    x_values = np.array(x_values)
    y_values = np.array(y_values)

    # Use numpy's polyfit to perform linear regression and get the slope and intercept
    slope, intercept = np.polyfit(x_values, y_values, 1)

    # Retourner uniquement la pente et l'ordonnée à l'origine
    return slope, intercept


def calculate_t_test(field1, field2, s1, s2, n1, n2, significance):
    # Convertir les valeurs en nombres
    field1 = float(field1)
    field2 = float(field2)
    s1 = float(s1)
    s2 = float(s2)
    n1 = int(n1)
    n2 = int(n2)

    # Calculer la statistique t
    t_stat, p_value = stats.ttest_ind_from_stats(mean1=field1, std1=s1, nobs1=n1, mean2=field2, std2=s2, nobs2=n2)

    # Tester l'hypothèse nulle
    if p_value < significance:
        hypothesis_result = "Reject the null hypothesis"
    else:
        hypothesis_result = "Fail to reject the null hypothesis"

    return t_stat, p_value, hypothesis_result


def calculate_t_test2(field, tTestmi, sigma, n, significance):
    # Convertir les valeurs en nombres
    field = float(field)
    sigma = float(sigma)
    n = int(n)
    significance = float(significance)
    tTestmi = float(tTestmi)

    # Calculer le t-test
    t_statistic = (field - tTestmi) / (sigma / np.sqrt(n))

    # Calculer la p-value pour le test bilatéral
    p_value_two_sided = t.sf(abs(t_statistic), df=n - 1) * 2

    # Interpréter les résultats du test
    hypothesis_result_two_sided = "Hypothesis rejected: The sample mean is significantly different from the specified mean." if p_value_two_sided < significance else "Hypothesis not rejected: There is no significant difference between the sample mean and the specified mean."

    # Retourner les résultats du test sous forme de dictionnaire
    return {
        't_statistic': t_statistic,
        'p_value_two_sided': p_value_two_sided,
        'hypothesis_result_two_sided': hypothesis_result_two_sided,
    }


def test_traitement(request):
    if request.method == 'GET':
        test_type = request.GET.get('testType')
        if test_type:
            # Récupérer les paramètres communs à tous les tests
            significance = float(request.GET.get('significance', 0.05))

            if test_type == 'tTest':
                # Récupérer les paramètres spécifiques au t-test
                field1 = request.GET.get('tTestField1')
                field2 = request.GET.get('tTestField2')
                s1 = request.GET.get('tTestS1')
                s2 = request.GET.get('tTestS2')
                n1 = request.GET.get('tTestN1')
                n2 = request.GET.get('tTestN2')

                t_stat, p_value, hypothesis_result = calculate_t_test(field1, field2, s1, s2, n1, n2, significance)

                # Construire la réponse JSON avec chaque résultat dans des phrases distinctes
                result_json = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'hypothesis_result': hypothesis_result,
                    'formula': f"t = (X̄1 - X̄2) / sqrt(s1^2/n1 + s2^2/n2)"
                }

                return JsonResponse(result_json)

            elif test_type == 'zTest':
                # Récupérer les paramètres spécifiques au z-test
                field = request.GET.get('zTestField')
                sigma = request.GET.get('zTestSigma')
                n = request.GET.get('zTestN')
                zTestmi = request.GET.get('zTestmi')
                z_test_results = calculate_z_test(field, zTestmi, sigma, n, significance)

                # Extraire chaque résultat pour l'affichage
                z_statistic_result = z_test_results['z_statistic']
                p_value_two_sided_result = z_test_results['p_value_two_sided']
                p_value_left_result = z_test_results['p_value_left']
                p_value_right_result = z_test_results['p_value_right']
                hypothesis_result_two_sided = z_test_results['hypothesis_result_two_sided']
                hypothesis_result_left = z_test_results['hypothesis_result_left']
                hypothesis_result_right = z_test_results['hypothesis_result_right']

                # Construire la réponse JSON avec chaque résultat dans des phrases distinctes
                result_json = {
                    'z_statistic': z_statistic_result,
                    'p_value_two_sided': p_value_two_sided_result,
                    'p_value_left': p_value_left_result,
                    'p_value_right': p_value_right_result,
                    'hypothesis_result_two_sided': hypothesis_result_two_sided,
                    'hypothesis_result_left': hypothesis_result_left,
                    'hypothesis_result_right': hypothesis_result_right,
                    'formula': f"Z = (X̄ - μ) / (σ/ √n)"
                }

                return JsonResponse(result_json)

            elif test_type == 'tTest2':
                # Récupérer les paramètres spécifiques au t-test
                field = request.GET.get('tTestField2')
                sigma = request.GET.get('tTestSigma2')
                n = request.GET.get('testTestN2')
                tTestmi = request.GET.get('tTestmi2')
                t_test_results = calculate_t_test2(field, tTestmi, sigma, n, significance)

                # Extraire chaque résultat pour l'affichage
                t_statistic_result = t_test_results['t_statistic']
                p_value_two_sided_result = t_test_results['p_value_two_sided']
                hypothesis_result_two_sided = t_test_results['hypothesis_result_two_sided']

                # Construire la réponse JSON avec chaque résultat dans des phrases distinctes
                result_json = {
                    't_statistic': t_statistic_result,
                    'p_value_two_sided': p_value_two_sided_result,
                    'hypothesis_result_two_sided': hypothesis_result_two_sided,
                    'formula': f"t = (X̄ - μ) / (σ/ √n)"
                }

                return JsonResponse(result_json)

            elif test_type == 'linearRegression':
                x_values_str = request.GET.get('linearRegressionX', '')
                y_values_str = request.GET.get('linearRegressionY', '')

                x_values = [float(value) for value in x_values_str.split()]
                y_values = [float(value) for value in y_values_str.split()]

                # Appeler la fonction calculate_linear_regression
                slope, intercept = calculate_linear_regression(x_values, y_values)

                # Créer un graphique de dispersion avec la ligne de régression
                plt.scatter(x_values, y_values, label='Data points')
                plt.plot(x_values, slope * np.array(x_values) + intercept, color='red', label='Regression line')
                plt.xlabel('Variable indépendante (X)')
                plt.ylabel('Variable dépendante (Y)')
                plt.legend()

                # Convertir le graphique en image
                image_stream = io.BytesIO()
                plt.savefig(image_stream, format='png')
                image_stream.seek(0)

                # Encoder l'image en base64 pour l'inclure dans la réponse JSON
                image_data = base64.b64encode(image_stream.read()).decode('utf-8')

                # Fermer le graphique
                plt.close()

                # Retourner l'image en réponse JSON, ainsi que la pente et l'ordonnée à l'origine
                return JsonResponse({'image_path': image_data, 'slope': slope, 'intercept': intercept})

            else:
                return JsonResponse({'error': 'Invalid test type'})

        else:
            return JsonResponse({'error': 'Invalid test type'})
    else:
        return JsonResponse({'error': 'Invalid request method'})
    return render(request, 'testes.html')


def tests(request):
    return render(request, 'testes.html')