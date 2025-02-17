"""Create HTML interactive plots of best fit spectra with dataset navigation"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import html, dcc
import dash.dependencies as dd
from dash.exceptions import PreventUpdate
import os
import pathlib
from retrieval_base.retrieval import Retrieval
import retrieval_base.auxiliary_functions as af
from retrieval_base.config import Config

path = af.get_path(return_pathlib=True)
path_figures = pathlib.Path('/home/dario/phd/twa2x_paper/figures')

config_file = 'config_jwst.txt'
target = 'TWA28'
# run = None
run = 'lbl11_G2G3_fastchem_GP_0'
w_set='NIRSpec'

runs = dict(
    # TWA27A=['lbl11_G1G2G3_fastchem_0'],
    TWA28=[
        ('lbl11_G2G3_fastchem_GP_0', 'G2+G3 (GP)'), 
        # ('lbl11_G2G3_fastchem_0', 'G2+G3'),
        ],
    )

colors = dict(TWA28={'data':'black', 
                     'model':['brown', 'darkgreen', 'darkblue'], 
                     'crires': 'orange'},
              TWA27A={'data':'#733b27',
                      'model':['#0a74da'],
                      })


def load_data(target, run):
    cwd = os.getcwd()
    if target not in cwd:
        os.chdir(f'{path}/{target}')
        print(f'Changed directory to {target}')

    conf = Config(path=path, target=target, run=run)(config_file)        
        
    m_spec = af.pickle_load(f'{conf.prefix}data/bestfit_m_spec_NIRSpec.pkl')
    m_spec.flux = m_spec.flux.squeeze()
    d_spec = af.pickle_load(f'{conf.prefix}data/d_spec_NIRSpec.pkl')
    
    Cov = af.pickle_load(f'{conf.prefix}data/bestfit_Cov_NIRSpec.pkl')
    LogLike = af.pickle_load(f'{conf.prefix}data/bestfit_LogLike_NIRSpec.pkl')
    err = np.nan * np.ones_like(d_spec.flux)
    
    for i in range(d_spec.n_orders):
        for j in range(d_spec.n_dets):
            # mask_i = d_spec.mask_isfinite[i,]
            mask_ij = d_spec.mask_isfinite[i,j]

            if Cov is not None:
                err_ij = Cov[i,j].get_err(mask=mask_ij)
            else:
                err_ij = d_spec.err[i,j]
                    
            beta_ij = LogLike.beta[i,j]
            err_ij *= beta_ij # optimal uncertainty scaling
            err[i,j,:] = err_ij
        
    flux_factor = conf.config_data['NIRSpec'].get('flux_unit_factor', 1.0)
    
    d_spec.err = err
    d_spec.squeeze()
    m_spec.flux.squeeze()

    m_spec.flux /= flux_factor
    d_spec.flux /= flux_factor
    d_spec.err /= flux_factor
    return d_spec, m_spec

def create_dash_app(d_spec, m_spec):
    """Create Dash app with dataset navigation"""
    app = dash.Dash(__name__)
    
    n_datasets = len(d_spec.wave)
    
    app.layout = html.Div([
        html.H1("Interactive Spectrum Viewer"),
        html.Div([
            html.Button("←", id='prev-button', n_clicks=0),
            html.Span(id='dataset-indicator', style={'margin': '0 20px'}),
            html.Button("→", id='next-button', n_clicks=0),
        ], style={'textAlign': 'center', 'margin': '20px'}),
        dcc.Graph(id='spectrum-plot'),
        dcc.Store(id='current-dataset', data=0),
        html.Button("Save Plot as HTML", id='save-button', n_clicks=0),
        dcc.Download(id='download')
    ])

    @app.callback(
        [dd.Output('spectrum-plot', 'figure'),
         dd.Output('dataset-indicator', 'children'),
         dd.Output('current-dataset', 'data')],
        [dd.Input('prev-button', 'n_clicks'),
         dd.Input('next-button', 'n_clicks')],
        [dd.State('current-dataset', 'data')]
    )
    def update_plot(prev_clicks, next_clicks, current_idx):
        ctx = dash.callback_context
        if not ctx.triggered:
            idx = 0
        else:
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            if button_id == 'prev-button':
                idx = (current_idx - 1) % n_datasets
            elif button_id == 'next-button':
                idx = (current_idx + 1) % n_datasets
            else:
                idx = current_idx

        # Create figure with secondary y-axis
        fig = make_subplots(rows=2, cols=1, 
                           shared_xaxes=True,
                           vertical_spacing=0.1,
                           subplot_titles=('Spectrum', 'Residuals'),
                           row_heights=[0.8, 0.2])
        
        # Add lower bound of uncertainty first
        fig.add_trace(
            go.Scatter(
                x=d_spec.wave[idx],
                y=d_spec.flux[idx] - d_spec.err[idx],
                mode='lines',
                fill=None,
                line=dict(width=1),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=d_spec.wave[idx],
                y=d_spec.flux[idx] + d_spec.err[idx],
                mode='lines',
                fill='tonexty',
                line=dict(width=3),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=1, col=1
        )

        
        # Add data points and line
        # fig.add_trace(
        #     go.Scatter(
        #         x=d_spec.wave[idx], 
        #         y=d_spec.flux[idx],
        #         name='Data', 
        #         mode='lines+markers',
        #         line=dict(color='black', width=1),
        #         marker=dict(size=5)
        #     ),
        #     row=1, col=1
        # )
        
        # # Add model trace
        # fig.add_trace(
        #     go.Scatter(
        #         x=d_spec.wave[idx], 
        #         y=m_spec.flux[idx],
        #         name='Model', 
        #         mode='lines',
        #         line=dict(color='brown', width=1)
        #     ),
        #     row=1, col=1
        # )
        
        # # Add blackbody
        # fig.add_trace(
        #     go.Scatter(x=d_spec.wave[idx], y=m_spec.flux_bb[idx],
        #               name='Blackbody', line=dict(color='brown', width=1, dash='dash')),
        #     row=1, col=1
        # )
        
        # Add residuals
        res = d_spec.flux[idx] - m_spec.flux[idx]
        fig.add_trace(
            go.Scatter(x=d_spec.wave[idx], y=res,
                      name='Residuals', line=dict(color='brown', width=1)),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            xaxis2_title="Wavelength (nm)",
            yaxis_title="Flux (erg/s/cm²/nm)",
            yaxis2_title="Data - Model"
        )
        
        # Add zero line to residuals
        fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="gray", row=2, col=1)
        
        return fig, f'Dataset {idx + 1} of {n_datasets}', idx

    @app.callback(
        dd.Output('download', 'data'),
        dd.Input('save-button', 'n_clicks'),
        dd.State('spectrum-plot', 'figure'),
        prevent_initial_call=True
    )
    def save_plot(n_clicks, figure):
        """Save the current plot as an HTML file"""
        import plotly.io as pio
        filename = f"{path_figures}/interactive_spectrum_{n_clicks}.html"
        pio.write_html(figure, filename=filename)
        print(f'Saved interactive plot to {filename}')
        return dcc.send_file(filename)

    return app

# Example usage
if __name__ == "__main__":
    d_spec, m_spec = load_data(target, run)
    app = create_dash_app(d_spec, m_spec)
    app.run_server(debug=True)
    print(f'Interactive plot saved to {path_figures}/interactive_spectrum.html')