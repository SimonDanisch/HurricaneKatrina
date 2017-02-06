using ArrayFire

function kmeans(in::AbstractArray, init, image::AbstractString, img::AbstractArray, iter::Integer = 10)
    n = size(in, 1)
    d = size(in, 3)
    k = length(init)
    #data = in * 0.0
    mini = minimum(in, 1)
    maxi = maximum(in, 1)

    #mini  = reshape(mini, 1, 1, d)
    #maxi  = reshape(maxi, 1, 1, d)
    mini  = constant(mini[1], 1, 1, d)
    maxi  = constant(maxi[1], 1, 1, d)
    data  = (in - mini) ./ maxi
    init2 = Array{Float32, 3}(1, k, d)
    for i = 1:k
        init2[1, i, 1] = comp1(init[i])
        init2[1, i, 2] = comp2(init[i])
        init2[1, i, 3] = comp3(init[i])
    end
    means = AFArray{Float32, 3}(init2)

    curr_clusters = constant(0, size(data, 1)) - 1
    prev_clusters = constant(0, size(data, 1)) - 1

    for i = 1:iter
        prev_clusters = curr_clusters
        curr_clusters = clusterize(data, means)
        num_changed = countnz(prev_clusters .!= curr_clusters)
        @show(num_changed)
        if num_changed < (n/1000) + 1
            break
        end
        means = new_means(data, curr_clusters, k)
        @show i
    end

    means = maxi .* means + mini
    clusters = prev_clusters

    means, clusters
end

function clusterize(data::AbstractArray, means::AbstractArray)
    dists = distance(data, means)
    val, idx = minidx(dists, 2)
    idx
end

function distance(data::AbstractArray, means::AbstractArray)
    n = size(data, 1)
    k = size(means, 2)
    data2 = repeat(data, outer = [1,k,1])
    means2 = repeat(means, outer = [n,1,1])
    sum(abs(data2 - means2), 3)
end

function new_means(data::AbstractArray, clusters, k::Integer)
    d = size(data, 3)
    means = constant(0.0f0, 1, k, d)
    clustersd = repeat(clusters, outer = [1, 1, d])
    for i = 1:size(means,2)
        means[:, i, :] = sum(data .* (clusters .== i), 1) ./ (sum(clusters .== i, 1) + Float32(1e-5))
    end
    means
end
