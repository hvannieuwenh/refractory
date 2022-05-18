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
function LogLike(Param, Time_Q, State)
    p = [];
    # State needs to be adjusted here, and also in for i in ... loop, since it should be only the states in the list from 1 to tau_star = t1+t2
    summation = 0;
    for i in 2:length(Time_Q) # note this is starting from 2 - julia indexes from 1, so in line 97 there is an error on the first loop when trying to call lam_QA[0]
        Param=temp;
        lam_QA = (tanh.((Param[1] .+ Param[2]*(State' .- 1) )./2).+1)./2;
        lam_RQ = (tanh(Param[3]/2)+1)/2;

        tau_star = Time_Q[i]
        tau_tilda = (i-1)*del_t
        K=0;
        for j in 1:((tau_star-tau_tilda)/del_t)
            K = K + del_t * lam_QA[j*del_t+tau_tilda];
        end

        # floor to turn the float indices into integers
        term_1 = exp(-1*(i*del_t*lam_RQ + K));
        term_2_num = (1 - exp(del_t*(lam_RQ - lam_QA[floor(Int32,tau_tilda)])));
        term_2_denom = (lam_QA[floor(Int32,tau_tilda)] - lam_RQ);

        to_sum = term_1*(term_2_num/term_2_denom);
        summation = summation + to_sum;

        append!(p,(lam_RQ*lam_QA[floor(Int32,tau_star)]))
    end

    L=0;
    for i in 1:length(p)
        L = L + log.(p)

    end

    return -1 * L

end
