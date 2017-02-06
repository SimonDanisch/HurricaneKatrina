using FileIO, Colors, Images

function map_idx(f, data, a, b, d)
    for idx in eachindex(data)
        f(idx, data, a, b, d)
    end
end

@inline function euclidian(a::Colorant, b::Colorant)
    abs(comp1(a) - comp1(b)) +
    abs(comp2(a) - comp2(b)) +
    abs(comp3(a) - comp3(b))
end

@inline function label!(idx, data, means, labels, k)
    T = typeof(euclidian(first(means), first(data)))
    @inbounds begin
        val = data[idx]
        label = 0; mindist = typemax(T)
        for ki = 1:k
            dist = euclidian(means[ki], val)
            if dist < mindist
                mindist = dist
                label = ki
            end
        end
        labels[idx] = label
    end
end
@inline function meancount(vlk)
    v, label, k = vlk
    count = label == k
    v * count, Int32(count)
end
dotplus(ab0, ab1) = (ab1[1] + ab0[1], ab1[2] + ab0[2]) # could be .+ on 0.6
function shiftmeans!(data, means, labels)
    for k = 1:length(means)
        sum, len = mapreduce(
            meancount, dotplus,
            zip(data, labels, repeated(k))
        )
        means[k] = sum / (len + 1e-5)
    end
end
function clustersmoved(prev_means, curr_means)
    mapreduce(
        ab-> Int32(ab[1] != ab[2]), +,
        zip(prev_means, curr_means)
    )
end
function kmeans{T <: AbstractArray}(A::T, initialmeans, iter = 10)
    means = copy(initialmeans)
    n = length(A); CT = eltype(T); ET = eltype(CT); k = length(means)
    labels = map(x-> UInt8(0), A)
    prevlabels = copy(labels)

    for i = 1:iter
        prevlabels[:] = labels
        map_idx(label!, A, means, labels, k)
        num_changed = clustersmoved(prevlabels, labels)
        if num_changed < (n / 1000) + 1
            return means, labels
        end
        shiftmeans!(A, means, labels)
        @show i
    end
    means, labels
end
