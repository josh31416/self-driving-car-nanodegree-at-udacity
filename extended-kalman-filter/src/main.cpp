#include "json.hpp"
#include "measurement_package.h"
#include "fusion_ekf.h"
#include <uWS/uWS.h>
#include <iostream>

using std::string;
using std::vector;
using std::cout;
using std::endl;
using json = nlohmann::json;
using Eigen::VectorXd;


string getData(string s);
MeasurementPackage parseJsonSensorMeasurement(string sensor_measurement);
VectorXd calculateRMSE(const vector<VectorXd> &estimations,
                       const vector<VectorXd> &ground_truths);


int main() {
    uWS::Hub h;
    const int port = 4567;

    FusionEKF fusion_ekf;
    vector<VectorXd> estimations;
    vector<VectorXd> ground_truths;

    h.onMessage([&fusion_ekf, &estimations, &ground_truths]
        (uWS::WebSocket<uWS::SERVER> ws, char *message, size_t length,
        uWS::OpCode opCode) {
            // "42" -> websocket message (4) event (2).
            if (length && length > 2 && message[0] == '4' && message[1] == '2') {
                string data = getData(string(message));
                if (data != "") {
                    auto json_data = json::parse(data);
                    if (json_data[0].get<string>() == "telemetry") {
                        MeasurementPackage measurement_package = parseJsonSensorMeasurement(json_data[1]["sensor_measurement"]);
                        // process measurement with kalman filter
                        fusion_ekf.ProcessMeasurement(measurement_package);
                        // calculate RMSE
                        estimations.push_back(fusion_ekf.ekf_.x_);
                        ground_truths.push_back(measurement_package.ground_truth_);
                        VectorXd RMSE = calculateRMSE(estimations, ground_truths);
                        json msgJson;
                        msgJson["estimate_x"] = fusion_ekf.ekf_.x_(0);
                        msgJson["estimate_y"] = fusion_ekf.ekf_.x_(1);
                        msgJson["rmse_x"] =  RMSE(0);
                        msgJson["rmse_y"] =  RMSE(1);
                        msgJson["rmse_vx"] = RMSE(2);
                        msgJson["rmse_vy"] = RMSE(3);
                        string res = "42[\"estimate_marker\","+msgJson.dump()+"]";
                        ws.send(res.data(), res.length(), uWS::OpCode::TEXT);
                    }
                } else {
                    string res = "42[\"manual\",{}]";
                    ws.send(res.data(), res.length(), uWS::OpCode::TEXT);
                }
            }
        }
    );

    h.onConnection([&h]
        (uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
            cout << "Websocket connected." << endl;
        }
    );

    h.onDisconnection([&h]
        (uWS::WebSocket<uWS::SERVER>, int code, char *message,
        size_t length) {
            cout << "Websocket disconnected." << endl;
        }
    );

    if (h.listen(port)) {
        cout << "Listening on port " << port << endl;
    } else {
        cout << "Port " << port << " is not available" << endl;
        return -1;
    }

    h.run();
}


string getData(string s) {
    if (s.find("null") != string::npos) {
        return "";
    } else {
        auto b1 = s.find_first_of("[");
        auto b2 = s.find_first_of("]");
        if (b1 != string::npos && b2 != string::npos) {
            return s.substr(b1, b2 - b1 + 1);
        }
        return "";
    }
}


MeasurementPackage parseJsonSensorMeasurement(string sensor_measurement) {
    MeasurementPackage measurement_package;
    std::istringstream iss(sensor_measurement);

    long long timestamp;
    string sensor_type;
    iss >> sensor_type;
    if (sensor_type.compare("L") == 0) {
        measurement_package.sensor_type_ = MeasurementPackage::LASER;
        measurement_package.raw_measurements_ = VectorXd(2);
        float px;
        iss >> px;
        float py;
        iss >> py;
        measurement_package.raw_measurements_ << px, py;
        iss >> timestamp;
        measurement_package.timestamp_ = timestamp;
    } else if (sensor_type.compare("R") == 0) {
        measurement_package.sensor_type_ = MeasurementPackage::RADAR;
        measurement_package.raw_measurements_ = VectorXd(3);
        float ro;
        iss >> ro;
        float theta;
        iss >> theta;
        float ro_dot;
        iss >> ro_dot;
        measurement_package.raw_measurements_ << ro, theta, ro_dot;
        iss >> timestamp;
        measurement_package.timestamp_ = timestamp;
    }

    measurement_package.ground_truth_ = VectorXd(4);
    float x_gt;
    iss >> x_gt;
    float y_gt;
    iss >> y_gt;
    float vx_gt;
    iss >> vx_gt;
    float vy_gt;
    iss >> vy_gt;
    measurement_package.ground_truth_ << x_gt, y_gt, vx_gt, vy_gt;

    return measurement_package;
}


VectorXd calculateRMSE(const vector<VectorXd> &estimations,
                       const vector<VectorXd> &ground_truths) {
    VectorXd rmse(4);
    rmse << 0, 0, 0, 0;

    // check validity of data
    if (estimations.size() != ground_truths.size() || estimations.size() == 0) {
        cout << "Invalid estimation or ground_truth data" << endl;
        return rmse;
    }

    // accumulate
    for (int i=0; i < estimations.size(); ++i) {
        VectorXd residual = estimations[i] - ground_truths[i];
        residual = residual.array()*residual.array();
        rmse += residual;
    }
    // mean
    rmse = rmse/estimations.size();
    // square root
    rmse = rmse.array().sqrt();

    return rmse;
}