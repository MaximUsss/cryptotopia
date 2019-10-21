# index page
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from server import app
from flask_login import logout_user, current_user
from views import login, login_fd, logout
import plotly.graph_objs as go
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.model_selection import train_test_split
from crypto import CCurrency

curr = CCurrency()


def format_dropdown_options(coins):
  labels = []
  for coin in coins:
    label_dict = {}
    label_dict['label'] = coins[coin]
    label_dict['value'] = coin
    labels.append(label_dict)
  return labels


header = html.Div(
    className='header',
    children=html.Div(
        className='container-width',
        style={'height': '100%'},
        children=[
            html.Img(
                # id="logo",
                src='/assets/dash-logo-stripe.svg',
                className='logo'
            ),
            html.Div(className='links', children=[
                html.Div(id='user-name', className='link'),
                html.Div(id='logout', className='link')
            ])
        ]
    )
)

# dash components
app_layout = html.Div([
    # Setting the main title of the Dashboard
    html.H1("Cryptocurrency Data Analysis", style={"textAlign": "center"}),
    # Dividing the dashboard in tabs
    dcc.Tabs(id="tabs", children=[
        # Defining the layout of the first Tab
        dcc.Tab(label='Basic graphs', children=[
            html.Div([
                html.H1("Crypro Prices High vs Lows",
                        style={'textAlign': 'center'}),
                # Adding the first dropdown menu and the subsequent time-series graph
                dcc.Dropdown(id='my-dropdown',
                             options=format_dropdown_options(curr.coins),
                             multi=True, value=['BTC'],
                             style={"display": "block", "marginLeft": "auto",
                                    "marginRight": "auto", "width": "60%"}),
                dcc.Graph(id='highlow'),
                html.H1("Cryptocurrency Market Volume",
                        style={'textAlign': 'center'}),
                # Adding the second dropdown menu and the subsequent time-series graph
                dcc.Dropdown(id='my-dropdown2',
                             options=format_dropdown_options(curr.coins),
                             multi=True, value=['BTC'],
                             style={"display": "block", "marginLeft": "auto",
                                    "marginRight": "auto", "width": "60%"}),
                dcc.Graph(id='volume')
            ], className="container"),
        ]),
        dcc.Tab(label='Machine Learning', children=[
            html.Div([html.H1("Machine Learning", style={"textAlign": "center"}), html.H2("ARIMA Time Series Prediction", style={"textAlign": "left"}),
                      dcc.Dropdown(id='my-dropdowntest', options=format_dropdown_options(curr.coins), value="BTC",
                                   style={"display": "block", "marginLeft": "auto", "marginRight": "auto", "width": "50%"}),
                      dcc.RadioItems(id="radiopred", value="high", labelStyle={'display': 'inline-block', 'padding': 10},
                                     options=[{'label': "High", 'value': "high"}, {'label': "Low", 'value': "low"},
                                              {'label': "Volume", 'value': "volumeto"}], style={'textAlign': "center", }),
                      dcc.Graph(id='traintest'),
                      dcc.Graph(id='preds')
                      ], className="container")
        ])
    ])
])


@app.callback(Output('highlow', 'figure'),
              [Input('my-dropdown', 'value')])
def update_hight_low_graph(selected_dropdown):
  dropdown = curr.coins
  trace1 = []
  trace2 = []
  for coin in selected_dropdown:
    df = curr.get_df(1504435200, 1534435200, coin)
    trace1.append(go.Scatter(x=df["time"],
                             y=df["high"],
                             mode='lines', opacity=0.7,
                             name='High %s' % dropdown[coin], textposition='bottom center'))
    trace2.append(go.Scatter(x=df["time"],
                             y=df["low"],
                             mode='lines', opacity=0.6,
                             name='Low %s' % dropdown[coin], textposition='bottom center'))
  traces = [trace1, trace2]
  data = [val for sublist in traces for val in sublist]
  figure = {'data': data,
            'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1',
                                          '#FF7400', '#FFF400', '#FF0056'],
                                height=600,
                                title="High and Low Prices for %s Over Time" % ', '.join(
                str(dropdown[i]) for i in selected_dropdown),
                xaxis={"title": "Date",
                       'rangeselector': {'buttons': list([{'count': 1, 'label': '1M',
                                                           'step': 'month',
                                                           'stepmode': 'backward'},
                                                          {'count': 6, 'label': '6M',
                                                           'step': 'month',
                                                           'stepmode': 'backward'},
                                                          {'step': 'all'}])},
                       'rangeslider': {'visible': True}, 'type': 'date'},
                yaxis={"title": "Price (USD)"})}
  return figure


@app.callback(Output('volume', 'figure'),
              [Input('my-dropdown2', 'value')])
def update_volume_graph(selected_dropdown_value):
  dropdown = curr.coins
  trace1 = []
  for coin in selected_dropdown_value:
    df = curr.get_df(1504435200, 1534435200, coin)
    trace1.append(go.Scatter(x=df["time"],
                             y=df["volumeto"],
                             mode='lines', opacity=0.7,
                             name='Volume %s' % dropdown[coin], textposition='bottom center'))
  traces = [trace1]
  data = [val for sublist in traces for val in sublist]
  figure = {'data': data,
            'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1',
                                          '#FF7400', '#FFF400', '#FF0056'],
                                height=600,
                                title="Market Volume for %s Over Time" % ', '.join(
                str(dropdown[i]) for i in selected_dropdown_value),
                xaxis={"title": "Date",
                       'rangeselector': {'buttons': list([{'count': 1, 'label': '1M',
                                                           'step': 'month',
                                                           'stepmode': 'backward'},
                                                          {'count': 6, 'label': '6M',
                                                           'step': 'month',
                                                           'stepmode': 'backward'},
                                                          {'step': 'all'}])},
                       'rangeslider': {'visible': True}, 'type': 'date'},
                yaxis={"title": "Transactions Volume"})}
  return figure


@app.callback(Output('traintest', 'figure'),
              [Input('my-dropdowntest', 'value'), Input("radiopred", "value")])
def update_ml_graph(coin, radioval):
  dropdown = curr.coins
  df = curr.get_df(1504435200, 1534435200, coin)
  radio = {"high": "High Prices",
           "low": "Low Prices", "volumeto": "Market Volume"}
  trace1 = []
  trace2 = []
  msk = np.random.rand(len(df)) < 0.8
  train_data = df[msk]
  test_data = df[~msk]
  if (coin is None):
    trace1.append(
        go.Scatter(x=[0], y=[0],
                   mode='markers', opacity=0.7, textposition='bottom center'))
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=['#FF7400', '#FFF400', '#FF0056'],
                                  height=600, title=f"{radio[radioval]}",
                                  paper_bgcolor='rgba(0,0,0,0)',
                                  plot_bgcolor='rgba(0,0,0,0)')}
  else:
    trace1.append(go.Scatter(x=train_data['time'], y=train_data[radioval], mode='lines',
                             opacity=0.7, name=f'Training Set', textposition='bottom center'))
    trace2.append(go.Scatter(x=test_data['time'], y=test_data[radioval], mode='lines',
                             opacity=0.6, name=f'Test Set', textposition='bottom center'))
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                                  height=600, title=f"{radio[radioval]} Train-Test Sets for {dropdown[coin]}",
                                  xaxis={"title": "Date",
                                         'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 'step': 'month', 'stepmode': 'backward'},
                                                                            {'count': 6, 'label': '6M',
                                                                             'step': 'month', 'stepmode': 'backward'},
                                                                            {'step': 'all'}])},
                                         'rangeslider': {'visible': True}, 'type': 'date'}, yaxis={"title": "Price (USD)"},
                                  paper_bgcolor='rgba(0,0,0,0)',
                                  plot_bgcolor='rgba(0,0,0,0)')}
  return figure


@app.callback(Output('preds', 'figure'),
              [Input('my-dropdowntest', 'value'), Input("radiopred", "value")])
def update_graph(coin, radioval):
  dropdown = curr.coins
  df = curr.get_df(1504435200, 1534435200, coin)
  radio = {"high": "High Prices",
           "low": "Low Prices", "volumeto": "Market Volume"}
  trace1 = []
  trace2 = []
  if (coin is None):
    trace1.append(
        go.Scatter(x=[0], y=[0],
                   mode='markers', opacity=0.7, textposition='bottom center'))
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=['#FF7400', '#FFF400', '#FF0056'],
                                  height=600, title=f"{radio[radioval]}",
                                  paper_bgcolor='rgba(0,0,0,0)',
                                  plot_bgcolor='rgba(0,0,0,0)')}
  else:
    msk = np.random.rand(len(df)) < 0.8
    train_data = df[msk]
    test_data = df[~msk]
    train_ar = train_data[radioval].values
    test_ar = test_data[radioval].values
    history = [x for x in train_ar]
    predictions = list()
    for t in range(len(test_ar)):
      model = ARIMA(history, order=(3, 1, 0))
      model_fit = model.fit(disp=0)
      output = model_fit.forecast()
      yhat = output[0]
      predictions.append(yhat)
      obs = test_ar[t]
      history.append(obs)
    error = mean_squared_error(test_ar, predictions)
    trace1.append(go.Scatter(x=test_data['time'], y=test_data[radioval], mode='lines',
                             opacity=0.6, name=f'Actual Series', textposition='bottom center'))
    trace2.append(go.Scatter(x=test_data['time'], y=np.concatenate(predictions).ravel(), mode='lines',
                             opacity=0.7, name=f'Predicted Series (MSE: {error})', textposition='bottom center'))
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', '#FF7400', '#FFF400', '#FF0056'],
                                  height=600, title=f"{radio[radioval]} ARIMA Predictions vs Actual for {dropdown[coin]}",
                                  xaxis={"title": "Date",
                                         'rangeselector': {'buttons': list([{'count': 1, 'label': '1M', 'step': 'month', 'stepmode': 'backward'},
                                                                            {'count': 6, 'label': '6M', 'step': 'month', 'stepmode': 'backward'},
                                                                            {'step': 'all'}])},
                                         'rangeslider': {'visible': True}, 'type': 'date'}, yaxis={"title": "Price (USD)"}, paper_bgcolor='rgba(0,0,0,0)',
                                  plot_bgcolor='rgba(0,0,0,0)')}
  return figure


app.layout = html.Div(
    [
        header,
        html.Div([
            html.Div(
                html.Div(id='page-content', className='content'),
                className='content-container'
            ),
        ], className='container-width'),
        dcc.Location(id='url', refresh=False)
    ]
)


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
  print(pathname)
  if pathname == '/':
    return login.layout
  elif pathname == '/login':
    return login.layout
  elif pathname == '/application':
    if current_user.is_authenticated:
      return app_layout
    else:
      return login_fd.layout
  elif pathname == '/logout':
    if current_user.is_authenticated:
      logout_user()
      return logout.layout
    else:
      return logout.layout
  else:
    return '404'


@app.callback(
    Output('user-name', 'children'),
    [Input('page-content', 'children')])
def cur_user(input1):
  if current_user.is_authenticated:
    return html.Div('Current user: ' + current_user.username)
    # 'User authenticated' return username in get_id()
  else:
    return ''


@app.callback(
    Output('logout', 'children'),
    [Input('page-content', 'children')])
def user_logout(input1):
  if current_user.is_authenticated:
    return html.A('Logout', href='/logout')
  else:
    return ''


if __name__ == '__main__':
  app.run_server(port=8080, debug=True)
