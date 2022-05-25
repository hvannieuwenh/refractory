using DelimitedFiles

using Statistics, Optim, SpecialFunctions, LinearAlgebra, LineSearches, Zygote, DataFrames, JLD

using IrrationalConstants:twoπ,halfπ,sqrtπ,sqrt2π,invπ,inv2π,invsqrt2,invsqrt2π,logtwo,logπ,log2π

using Plots

import ChainRulesCore


function LogLike(Param,Time_Q,State)
    
    L = 0;
    # State needs to be adjusted here, and also in for i in ... loop, since it should be only the states in the list from 1 to tau_star = t1+t2

    #state here is a 2d array. For each time_Q we have the state of each interval (first needs to be the initial state value)
    for i in 1:Time_Q # note this is starting from 2 - julia indexes from 1, so in line 97 there is an error on the first loop when trying to call lam_QA[0]
        lam_RQ = (tanh(Param[3]/2)+1)/2; #scalar
        lam_QA = (tanh.((Param[1] .+ Param[2]*(State[i]) )./2).+1)./2; #array


        #summation
        summation = 0
        K = 0
        for j in length(State[i]):-1:1
            t1 = exp(-j*lam_RQ - K)
            t2 = 1 - exp((lam_RQ - lam_QA[j]))
            t3 = lam_QA[j] - lam_RQ
            sol = t1*t2/t3
            summation = summation + sol
            K = K + lam_QA[j]
        end

        summation = lam_RQ*lam_QA[end].*summation

        #log likelihood calculation
        p = log(summation)
        L = L + p
    end

    return -1 * L
    #return summation
end

num_neurons=1000

sz=num_neurons;

    Params=zeros(Float64, 4);

    d1 = load("data.jld")["data"]
    d2 = load("interval.jld")["interval"]
    d2 = d2[1:end-100]

#=
    Time_A=readdlm("Data/Ta_"*string(1, base = 10, pad = 0)*".csv", ',');

    for i in 2:sz
        global Time_A
        try
        Time_A_append = readdlm("Data/Ta_"*string(i, base = 10, pad = 0)*".csv", ',');
        catch
        Time_A_append = []
        finally
        Time_A = vcat(Time_A, Time_A_append)
        end
    end
    =#

    #new stuff
    State = [d1[trunc(Int, l[1]):trunc(Int, l[2])] for l in d2]
    for (i1, i2) in enumerate(State)
        if State[i1][end] != 0
        State[i1][end] = State[i1][end] - 1
        end
    end

    Time_Q = size(d2)[1]
    #State = State[1:1]

    #initial guess
    ic = [-4.0, 0, 0.01]

    ic[2] = ic[2] / num_neurons
    ic[3] = (atanh(2*ic[3]) - 1)*2

    print(LogLike(ic, Time_Q, State))
    obj(Par)=LogLike(Par,Time_Q,State);
    function g!(G,x)
         G.=obj'(x)
    end


res=optimize(obj,g!,  ic,LBFGS(; m=5, linesearch=BackTracking(order=3)), Optim.Options(iterations=10,show_trace=true));

Params[1:3]=Optim.minimizer(res);
Params[2] = Params[2]*num_neurons
Params[3] = (tanh(Params[3]/2)+1)/2
Params[end]=0.8;

print(Params)


population = zeros(3,10000)

pi_A = 0.0
pi_Q = 0.0
pi_R = 1.0

population[:,1] = [pi_R, pi_Q, pi_A]'

for i in 2:size(population)[2]
    p_RQ = Params[3]
    p_QA = (tanh.((Params[1] .+ Params[2]*population[3,i-1])./2).+1)./2;
    p_AR = Params[4]

    del_pi_Q = (1 - population[3,i-1] - population[2,i-1])*p_RQ - population[2,i-1]*p_QA
    del_pi_A = population[2,i-1]*p_QA - population[3,i-1]*p_AR
    del_pi_R = 0-(del_pi_Q+del_pi_A)

    del = [del_pi_R, del_pi_Q, del_pi_A]

    population[:,i] = population[:,i-1].+(del)
end


