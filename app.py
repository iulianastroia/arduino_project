import dash_html_components as html
from flask import render_template, request

from facebook_prophet import calculate_facebook_prophet
from linear_regression import calculate_linear_regression, create_figure
from maindash import server, app
from homepage import homepage_layout, create_data
from polynomial_regression import calculate_polynomial_regression
from spline_regression import calculate_spline_regression
from table import table_layout


@server.route("/")
def timeseries():
    app.layout = homepage_layout()
    return app.index()

@server.route("/table")
def table():
    app.layout = table_layout()
    return app.index()

@server.route("/future")
def future():
    # app.layout = html.Div(id='dash-container', children='future')
    # return app.index()
    return render_template("future_forecast.html")


@server.route("/temperature_model")
def temperature_model():
    return render_template("prediction_models/Temperature.html")


@server.route("/humidity_model")
def humidity_model():
    return render_template("prediction_models/Humidity.html")


# selecting predictions
@server.route("/predictions", methods=["POST", "GET"])
def menu():
    if request.method == "POST":
        sensor_name = request.form["sensors"]
        select_prediction_id = request.form["select_prediction_id"]
        if sensor_name == "1":
            sensor = "Temperature"
        if sensor_name == "2":
            sensor = "Humidity"
        if select_prediction_id == "1":  # linear regression
            df = create_data()
            group_by_df, X_train, sensor_name, mse_value = calculate_linear_regression(df, sensor)
            linear_reg_fig = create_figure(group_by_df, X_train, sensor_name)
            print('prediction is linear', sensor)
            info_message = ("MSE is ") + str(mse_value)

            return render_template("predictions.html", error=info_message, prediction_plot_py=linear_reg_fig)
        if select_prediction_id == "2":  # polynomial regression
            df = create_data()
            polynomial_fig, minimum_mse_val, poly_grade_min_mse = calculate_polynomial_regression(df,
                                                                                                  sensor,
                                                                                                  7)
            info_message = "Best prediction grade is " + str(poly_grade_min_mse) + " for a MSE equal to " + str(
                minimum_mse_val)
            return render_template("predictions.html", error=info_message, prediction_plot_py=polynomial_fig)

        if select_prediction_id == "3":  # spline regression
            df = create_data()
            spline_fig, minimum_mse_val, spline_grade_min_mse = calculate_spline_regression(df, sensor)
            info_message = "Best prediction grade is " + str(spline_grade_min_mse) + " for a MSE equal to " + str(
                minimum_mse_val)
            return render_template("predictions.html", error=info_message, prediction_plot_py=spline_fig)

        if select_prediction_id == "4":  # facebook prophet
            df = create_data()

            try:
                prophet_fig, mse_value = calculate_facebook_prophet(df, sensor)
                info_message = "MSE is " + str(mse_value)
                return render_template("predictions.html", error=info_message, prediction_plot_py=prophet_fig)
            except TypeError:
                error = "Cannot calculate predictions."
                return render_template("predictions.html", error=error)

        print('sensor name is', sensor)
    return render_template("predictions.html")


if __name__ == "__main__":
    app.layout = html.Div(id='main-app', children='main app')
    server.run(debug=True)
