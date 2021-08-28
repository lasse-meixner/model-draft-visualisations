#For Heroku (synthesised from .ipynb)

import plotly.express as px
import dash_core_components as dcc
import dash_html_components as html
import dash
from dash.dependencies import Input, Output

import numpy as np
from sympy import *
import pandas as pd

a,m,si,E,L2,L1,L = symbols("W_a,W_m,s_i,E_m,L_{II},L_I,L")
pi_eq = symbols(pretty(pi)+"_eq")
pi_m = symbols(pretty(pi)+"_M")
s_eq = symbols("s\u0302")

Pi_eq = a/((a*si)+(1-si)*m)
Pi_M = E/(L1 + si*L2)
Pi_eq_func = lambdify([a,m,si],Pi_eq,"numpy")
Pi_M_func = lambdify([E,L1,L2,si],Pi_M,"numpy")

intersection = Eq(Pi_eq,Pi_M)
s_eq = solve(intersection,si)[0]
s_eq_func = lambdify([m,a,E,L,L1],s_eq.subs(L2,(L-L1)),"numpy")

endog = Eq(((a*L1-E*m)/(E*(a-m)-a*(L-L1)))*L,L1)

L1_sol0 = lambdify([m,a,E,L],solve(endog,L1)[0],"numpy")

L_m, U, s_t = symbols("L_m,U,sbar")
Pi_m = E/(L1*(1-s_t)+L*s_t)

L_m = L1*(1-Pi_m*s_t) + L*Pi_m*s_t
L_m_sol = L_m.subs(L1,L1_sol)
L_m_sol_func = lambdify([m,a,E,L,s_t],L_m_sol,"numpy")

UR = 1-(E/L_m_sol)
UR
unemployment_rate_func = lambdify([m,a,E,L,s_t],UR,"numpy")

#FB
L_m_FB = L+ E*(m/a) + (m*L)/((m-a)*s_t-m)
L_m_FB_func = lambdify([m,a,E,L,s_t],L_m_FB,"numpy")
UR_FB  = (L_m_FB - E)/L_m_FB
UR_FB_func = lambdify([m,a,E,L,s_t],UR_FB,"numpy")

#Dash App
import dash_bootstrap_components as dbc

app = Dash(__name__,external_stylesheets=[dbc.themes.GRID])
server = app.server


app.layout = html.Div([
    html.Div([
        dbc.Row([
            dbc.Col(dcc.Graph(id='graph'),align="start"),
            dbc.Col(dcc.Graph(id="graph2"),align="center"),
            dbc.Col(dcc.Graph(id="graph3"),align="end")
                ])
            ]),
    
    html.Div([
        dbc.Row([
            dbc.Col(
                html.Div(["Wm",
                    dcc.Input(
                        id='m_input',
                        value=12, type="number")
                        ])
                    ),
            dbc.Col(
                html.Div(["Wa:",
                    dcc.Input(
                        id="a_input",
                        value=4, type="number")
                        ])
                    ),
            dbc.Col(
                html.Div(["E:",
                    dcc.Input(
                        id="E_input",
                        value=50, type="number")
                        ])
                    )
                ]),
        
        dbc.Row([
            dbc.Col(
                html.Div(["L:",
                    dcc.Input(
                        id="L_input",
                        value=100, type="number")
                        ])
                    ),
            dbc.Col(
                html.Div(["S_true:",
                    dcc.Slider(
                        id="s_true_input", min=0, max=1, step=0.1,
                        value=0.5)     
                        ])
                    )
                ])
            ]),
])
    

@app.callback(
    Output("graph","figure"),
    [Input("m_input","value"),
    Input("a_input","value"),
    Input("E_input","value"),
    Input("L_input","value")])
def update_graph(m,a,E,L):
    s = np.linspace(0,1,1000)
    L1 = L1_sol0(m,a,E,L)
    y1 = Pi_eq_func(a,m,s)
    y2 = Pi_M_func(E,L1,L-L1,s)
    df = pd.DataFrame({"s_i":s,"Pi_eq":y1,"Pi_M":y2})
    fig = px.line(df,y=["Pi_eq","Pi_M"],x="s_i")
    fig.add_vline(x=s_eq_func(m,a,E,L,L1))
    return fig

@app.callback(
    Output("graph2","figure"),
    [Input("m_input","value"),
    Input("a_input","value"),
    Input("E_input","value"),
    Input("L_input","value"),
    Input("s_true_input","value")])
def update_ex_post_graph(m,a,E,L,s):
    bar = pd.DataFrame({"Title":["L_I","L_m","L_m_Fields"], "Solution": [L1_sol0(m,a,E,L),L_m_sol_func(m,a,E,L,s),L_m_FB_func(m,a,E,L,s)]})
    fig = px.bar(bar,x="Title",y="Solution")
    return fig

@app.callback(
    Output("graph3","figure"),
    [Input("m_input","value"),
    Input("a_input","value"),
    Input("E_input","value"),
    Input("L_input","value"),
    Input("s_true_input","value")])
def update_ex_post_graph(m,a,E,L,s):
    bar = pd.DataFrame({"Title":["Unemployment Rate","Unemployment Rate Fields"], "Solution": [unemployment_rate_func(m,a,E,L,s),UR_FB_func(m,a,E,L,s)]})
    fig = px.bar(bar,x="Title",y="Solution")
    return fig
    
if __name__ == '__main__':
    app.run_server()
