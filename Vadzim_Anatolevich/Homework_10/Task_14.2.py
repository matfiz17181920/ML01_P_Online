#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Блок импорта библиотек

import pandas as pd;
import plotly.graph_objects as go;

#Блок кода

odometry_file_path = 'odometry.csv';
odometry_data = pd.read_csv(odometry_file_path);
odometry_data.columns = odometry_data.columns.str.strip();
camera_positions = odometry_data[['x', 'y', 'z']].values;

fig = go.Figure();
fig.add_trace(go.Scatter3d(
    x = camera_positions[:, 0],
    y = camera_positions[:, 1],
    z = camera_positions[:, 2],
    mode = 'markers',
    marker = dict(size = 5, color = 'red'),
    name = 'Camera Positions'
    ));
fig.update_layout(
    title = 'Camera Positions in 3D Space',
    scene = dict(
        xaxis_title = 'X (m)',
        yaxis_title = 'Y (m)',
        zaxis_title = 'Z (m)'
    ));
fig.write_html('3d_scatter_plot.html');
fig.show();

