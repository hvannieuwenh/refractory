using Random

#define randomness
random_1 = MersenneTwister(1234)
random_2 = MersenneTwister(5678)

#generate test data for validation (inverse sampling)
function Generate_Test_Data(num_points, h_RQ, h_QA, lambda_AR)

    data = zeros(1,num_points);
    data_2 = zeros(1,num_points);

    for i in 1:num_points
        lambda_RQ = (tanh(h_RQ/2)+1)/2;
        A = -(1/lambda_RQ)*log(1-rand(random_1));

        lambda_QA = (tanh(h_QA/2)+1)/2;
        B = -(1/lambda_QA)*log(1-rand(random_2));

        data[i] = A + B;
    end

    for i in 1:num_points
        C = -(1/lambda_AR)*log(1-rand());
        data_2[i] = C;
    end

    return data', data_2'

end