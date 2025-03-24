import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RadioButtons

def draw_function(x, ylim=(-10, 40), draw_limits=(-50, 50)):

    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.35)  
    ax.set_ylim(ylim)
    ax.grid(True)

    y = np.zeros_like(x)
    line, = ax.plot(x, y)
    points = []

    x_snap_options = {
        'Off': 0,
        '1': 1,
        '10': 10,
        '25': 25
    }
    y_snap_options = x_snap_options.copy() 
    current_x_snap = x_snap_options['Off']
    current_y_snap = y_snap_options['Off']


    def onclick(event):
        if event.button == 1 and event.inaxes == ax:
            if len(points) < 2:
                x_val = event.xdata
                y_val = event.ydata
                if current_x_snap:
                    x_val = snap_to_grid(x_val, current_x_snap)
                if current_y_snap:
                    y_val = snap_to_grid(y_val, current_y_snap)
                points.append((x_val, y_val))
                if len(points) == 2:
                    draw_line()


    def draw_line():
        x1, y1 = points[0]
        x2, y2 = points[1]

        x_index1 = np.argmin(np.abs(x - x1))
        x_index2 = np.argmin(np.abs(x - x2))
        x_range = sorted([x_index1, x_index2])

        x_data_indices = np.where((x >= draw_limits[0]) & (x <= draw_limits[1]))[0]
        x_data_indices_in_range = np.intersect1d(x_data_indices, np.arange(x_range[0], x_range[1]+1))

        if len(x_data_indices_in_range) > 0:
            y_interp = np.interp(x[x_data_indices_in_range], [x1, x2], [y1, y2])
            y[x_data_indices_in_range] = y_interp

        line.set_data(x, y)
        fig.canvas.draw_idle()
        points.clear()

    def reset_plot(event):
        nonlocal y, points
        y = np.zeros_like(x)
        line.set_data(x, y)
        points = []
        fig.canvas.draw_idle()

    def snap_to_grid(value, factor):
        return round(value / factor) * factor

    def toggle_x_snap(label):
        nonlocal current_x_snap
        current_x_snap = x_snap_options[label]

    def toggle_y_snap(label):
        nonlocal current_y_snap
        current_y_snap = y_snap_options[label]


    fig.canvas.mpl_connect('button_press_event', onclick)

    ax_reset = plt.axes([0.7, 0.05, 0.1, 0.075])
    reset_button = Button(ax_reset, 'Reset')
    reset_button.on_clicked(reset_plot)

    ax_x_radio = plt.axes([0.81, 0.05, 0.15, 0.1])
    x_radio = RadioButtons(ax_x_radio, list(x_snap_options.keys()))
    x_radio.on_clicked(toggle_x_snap)

    ax_y_radio = plt.axes([0.81, 0.16, 0.15, 0.1]) # Positioned below x radio buttons
    y_radio = RadioButtons(ax_y_radio, list(y_snap_options.keys()))
    y_radio.on_clicked(toggle_y_snap)

    plt.show()
    return y


# Main part of the program
x = np.linspace(-100, 100, 20001)

y1 = draw_function(x)
print("First function drawn. Close the plot to draw the second function.")

y2 = draw_function(x)
print("Second function drawn.")


# --- Testing Section ---
def test_functions(func1_data, func2_data, x_values):
    """Tests the drawn functions by plotting them and calculating some statistics."""

    plt.plot(x_values, func1_data, label="Function 1")
    plt.plot(x_values, func2_data, label="Function 2")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Drawn Functions")
    plt.legend()
    plt.grid(True)


    # Example statistics (you can add more as needed)
    print("\n--- Function Statistics ---")
    print("Function 1 - Max:", np.max(func1_data))
    print("Function 1 - Min:", np.min(func1_data))
    print("Function 2 - Max:", np.max(func2_data))
    print("Function 2 - Min:", np.min(func2_data))


    plt.show()



def convolve_functions(func1_data, func2_data, x_values):
    if func1_data.ndim != 1 or func2_data.ndim != 1 or x_values.ndim !=1:
        raise ValueError("Input arrays must be 1-dimensional.")
    if len(func1_data) != len(func2_data) or len(func1_data) != len(x_values):
        raise ValueError("Input arrays must have the same length.")

    convolved_data = np.convolve(func1_data, func2_data, mode='full')
    x_step = x_values[1]-x_values[0]
    x_convolved = np.linspace(x_values[0] - x_values[-1] + x_values[0], x_values[-1] + x_values[-1]-x_values[0], len(convolved_data), endpoint=True)

    #Scaling factor to reduce the y-values
    scale_factor = x_step

    convolved_data_scaled = convolved_data * scale_factor

    return convolved_data_scaled, x_convolved

convolved_data, x_convolved = convolve_functions(y1, y2, x)

plt.figure(figsize=(10, 6))  # Adjust figure size as needed
plt.plot(x_convolved, convolved_data, label="Convolution Result")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Convolution of Function 1 and Function 2")
plt.legend()
plt.grid(True)
plt.show()

test_functions(y1, y2, x)