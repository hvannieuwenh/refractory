using DelimitedFiles

using Statistics, Optim, SpecialFunctions, LinearAlgebra, LineSearches, Zygote, DataFrames

using IrrationalConstants:twoπ,halfπ,sqrtπ,sqrt2π,invπ,inv2π,invsqrt2,invsqrt2π,logtwo,logπ,log2π

import ChainRulesCore

cd("/Users/Jolien/Documents/lcneuro/mea_github/");

function MaxCal_Refrac(num_neurons,lambda)

    # Inputs and Outputs

     #=
     Input is the number of neurons (160 for salamander retinal cells, 96 for new data) and lambda (can take to be 0.5 for now).
     Here the inputs are (one per neuron, coming from /Users/Jolien/Documents/lcneuro/mea/hc-8/output_arrays/Culture10DIV10/ folder):
     Time_A (the time intervals of each spike for neuron N)
     Time_Q (the time intervals between each spike for neuron N)
     State (the state of each neuron at the end of each quiescent period. The state of the neuron of interest N is always 1 here).
    The output Params is an N x N+2 matrix of maximum entropy parameters
            Params[i,j]=Kij, Params[i,i]=hi, Params[i,N+1]=lam_rq_i, Params[i,N+2]=lam_ar_i
     =#

    # Optimization
    sz=num_neurons;

    Params=zeros(Float64, 4);

    i = 1;

    println("Reading in data...")
    #read in DataFrames
    Time_Q=readdlm("/Users/Jolien/Documents/lcneuro/mea/hc-8/output_arrays/Culture10DIV10/Tqr_"*string(i, base = 10, pad = 0)*".csv", ',');
    Time_A=readdlm("/Users/Jolien/Documents/lcneuro/mea/hc-8/output_arrays/Culture10DIV10/Ta_"*string(i, base = 10, pad = 0)*".csv", ',');
    State=readdlm("/Users/Jolien/Documents/lcneuro/mea/hc-8/output_arrays/Culture10DIV10/S_"*string(i, base = 10, pad = 0)*".csv", ',');

    for i in 2:sz[1]
        Time_Q = vcat(Time_A, readdlm("/Users/Jolien/Documents/lcneuro/mea/hc-8/output_arrays/Culture10DIV10/Ta_"*string(i, base = 10, pad = 0)*".csv", ','));
        Time_A = vcat(Time_Q, readdlm("/Users/Jolien/Documents/lcneuro/mea/hc-8/output_arrays/Culture10DIV10/Tqr_"*string(i, base = 10, pad = 0)*".csv", ','));
        State = hcat(State, readdlm("/Users/Jolien/Documents/lcneuro/mea/hc-8/output_arrays/Culture10DIV10/S_"*string(i, base = 10, pad = 0)*".csv", ','));
    end

    println("Data in.")

    State = sum(State, dims=1)

    ic=zeros(Float64, 3);
    ic[end]=1;
    #initial guess
    ic = [0.0, 0.0, -4.0]


    obj(Par)=LogLike(Par, Time_Q, State);
    function g!(G,x)
     G.=obj'(x)
    end

    #=
    println("Objective defined, moving to minimization...")

    #optimize step uses automatic differentiation
    # BFGS estimates second derivative
    res=optimize(obj,g!,  ic,LBFGS(; m=5, linesearch=BackTracking(order=3)),Optim.Options(show_trace=true));  #slow
    println("Residual = ",res)

    Params[1:3]=Optim.minimizer(res);
    Params[end]=1/mean(Time_A);
    Params[2] = Params[2]*num_neurons
    Params[3] = (tanh(Params[3]/2)+1)/2

    #Params = [h, J, lambda_RQ?, lambda_AR = 1/mean(tau_AR) = 1/mean(Time_A)]
    return Params
    =#
end

del_t=1;

# param = [h, J, lambda]
function LogLike(Param,Time_Q,State,lambda,ii)
    
    L = 0;
    # State needs to be adjusted here, and also in for i in ... loop, since it should be only the states in the list from 1 to tau_star = t1+t2

    #state here is a 2d array. For each time_Q we have the state of each interval (first needs to be the initial state value)
    for i in 1:Time_Q # note this is starting from 2 - julia indexes from 1, so in line 97 there is an error on the first loop when trying to call lam_QA[0]
        lam_RQ = (tanh(Param[3]/2)+1)/2; #scalar
        lam_QA = (tanh.((Param[1] .+ Param[2]*(State[i]) )./2).+1)./2; #array

        #define K, array
        K = reverse(lam_QA)
        K = cumsum(K)
        K = reverse(K)
        K = circshift(K, -1)
        K[end] = 0   


        #summation
        summation = 0
        for k in 1:length(State[i])
            t1 = exp.(-i*lam_RQ .+ K[k])
            t2 = 1 - exp.((lam_RQ .- lam_QA[k]))
            t3 = lam_QA[k] .- lam_RQ
            sol = t1.*t2./t3
            summation = summation .+ sol
        end

        summation = lam_RQ*lam_QA[end].*summation

        #log likelihood calculation
        p = log(summation)
        L = L + p
    end

    #log likelihood calculation
    #p = log.(p)
    #L = sum(p)

    return -1 * L
end
