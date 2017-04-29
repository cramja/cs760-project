#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

VectorXd sgd_step(VectorXd weights, VectorXd x, int y, double delta)
{
    VectorXd gradient = weights * delta;

    if (weights.dot(x) * y < 1.0) {
        gradient -= (y * x);
    }
    
    return gradient;
}

VectorXd sgd(MatrixXd x, VectorXi y, int epoch, double learning_rate, double delta)
{

    int x_row_cnt = x.rows();
    int x_col_cnt = x.cols();

    VectorXd x_row;
    VectorXd weights = VectorXd::Zero(x_col_cnt);

    for (int i = 0; i < epoch; i++) {
        for (int j = 0; j < x_row_cnt; j++) {
            x_row = x.row(j);
            weights -= learning_rate * sgd_step(weights, x_row, y(j), delta);
        }
        cout << "Round: " << i << "\n" << weights << endl;
    }

    return weights;
}

VectorXd asgd(MatrixXd x, VectorXi y, int epoch, int average_time, double learning_rate, double delta)
{
    int x_row_cnt = x.rows();
    int x_col_cnt = x.cols();

    VectorXd x_row;
    VectorXd weights = VectorXd::Zero(x_col_cnt);
    VectorXd avg_weights = VectorXd::Zero(x_col_cnt);
    VectorXd new_a_update = VectorXd::Zero(x_col_cnt);

    int t = 0;
    double mu;

    for (int i = 0; i < epoch; i++) {
        for (int j = 0; j < x_row_cnt; j++) {
            x_row = x.row(j);
            t += 1;

            if (t <= average_time) {
                mu = 1;
            } else {
                mu = (double)1 / (t - average_time);
            }

            weights -= learning_rate * sgd_step(weights, x_row, y(j), delta);
            new_a_update = mu * (weights - avg_weights);
            avg_weights += new_a_update;
        }
    }

    return avg_weights;
}

VectorXd saga(MatrixXd x, VectorXi y, int epoch, double learning_rate, double delta)
{
    int x_row_cnt = x.rows();
    int x_col_cnt = x.cols();

    VectorXd x_row;
    VectorXd temp_new;
    VectorXd gradient;
    VectorXd old_gradient;

    VectorXd weights = VectorXd::Zero(x_col_cnt);
    VectorXd avg_gradient = VectorXd::Zero(x_col_cnt);
    MatrixXd all_gradients = MatrixXd::Zero(x_row_cnt, x_col_cnt);

    for (int i = 0; i < epoch; i++) {
        for (int j = 0; j < x_row_cnt; j++) {
            x_row = x.row(j);

            temp_new = weights;
            gradient = sgd_step(temp_new, x_row, y(j), delta);
            old_gradient = all_gradients.row(j);

            weights -= learning_rate * (gradient - old_gradient + avg_gradient);

            avg_gradient += (gradient - old_gradient) / x_row_cnt;
            all_gradients.row(i) = gradient;
        }
    }

    return weights;
}

VectorXd sag(MatrixXd x, VectorXi y, int epoch, double learning_rate, double delta)
{
    int x_row_cnt = x.rows();
    int x_col_cnt = x.cols();

    VectorXd x_row;
    VectorXd acc;
    VectorXd gradient;
    VectorXd last_gradient;

    VectorXd weights = VectorXd::Zero(x_col_cnt);

    for (int i = 0; i < epoch; i++) {
        for (int j = 0; j < x_row_cnt; j++) {
            x_row = x.row(j);

            gradient = sgd_step(weights, x_row, y(j), delta);
            acc = acc - last_gradient + gradient;
            last_gradient = gradient;
            weights -= (learning_rate / x_row_cnt) * acc;
        }
    }

    return weights;
}

VectorXd svrg(MatrixXd x, VectorXi y, int epoch, double learning_rate, double delta)
{

    int x_row_cnt = x.rows();
    int x_col_cnt = x.cols();

    VectorXd x_row;
    VectorXd avg_gradients;
    VectorXd sum_gradients;
    VectorXd weights_new;
    VectorXd weights_temp;

    VectorXd weights = VectorXd::Zero(x_col_cnt);

    for (int i = 0; i < epoch; i++) {
        weights_new = weights;
        sum_gradients = VectorXd::Zero(x_col_cnt);

        for (int j = 0; j < x_row_cnt; j++) {
            x_row = x.row(j);
            sum_gradients += sgd_step(weights_new, x_row, y(j), delta);
        }

        avg_gradients = sum_gradients / x_row_cnt;
        weights_temp = weights_new;

        for (int j = 0; j < x_row_cnt; j++) {
            x_row = x.row(j);
            weights_temp -= learning_rate * (sgd_step(weights_temp, x_row, y(j), delta)
                                                - sgd_step(weights_new, x_row, y(j), delta)
                                                + avg_gradients);
        }

        weights = weights_temp;
    }

    return weights;
}

VectorXd central_vr(MatrixXd x, VectorXi y, int epoch, double learning_rate, double delta)
{
    
    int x_row_cnt = x.rows();
    int x_col_cnt = x.cols();

    VectorXd acc;
    VectorXd x_row;
    VectorXd old_gradient;
    VectorXd new_gradient;
    VectorXd weights = VectorXd::Zero(x_col_cnt);
    VectorXd avg_gradient = VectorXd::Zero(x_col_cnt);
    MatrixXd all_gradients = MatrixXd::Zero(x_row_cnt, x_col_cnt);

    //TODO: Normally weight, gradients, average must be initialized with 1 epoch of SGD.

    for (int i = 0; i < epoch; i++) {
        acc = VectorXd::Zero(x_col_cnt);
        for (int j = 0; j < x_row_cnt; j++) {
            x_row = x.row(j);

            old_gradient = all_gradients.row(j);
            new_gradient = sgd_step(weights, x_row, y(j), delta);

            weights -= learning_rate * (new_gradient - old_gradient + avg_gradient);
            acc += new_gradient / x_row_cnt;
            all_gradients.row(i) = new_gradient;
        }
        avg_gradient = acc;
    }

    return weights;
}

int main()
{

    MatrixXd x(5,5);

    x << 1, 0, 0, 0, 0,
         0, 2, 0, 0, 0,
         0, 0, 3, 0, 0,
         0, 0, 0, 4, 0,
         0, 0, 0, 0, 5;
    
    VectorXi y(5);
    y << -1, 1, 1, -1, -1;

    int param = 5;
    VectorXd weights = VectorXd::Zero(param);

    int epoch = 10;
    double delta = 0.1;
    double learning_rate = 0.1;

    sgd(x, y, epoch, learning_rate, delta);
    /*
    for (int j = 0; j < 10; j++) {
        for (int i = 0; i < 5; i++) {
            VectorXd row = x.row(i);
            
            weights -= learning_rate * sgd_step(weights, row, y(i), delta);
        }
        cout << "Round: " << j << "\n" << weights << endl;
    }
    */

    MatrixXd all_gradients = MatrixXd::Zero(param, param);
    cout << "Matrix is :\n" << all_gradients << endl;
    return 0;
}
