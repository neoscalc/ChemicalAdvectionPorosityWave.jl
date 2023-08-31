
@inline function limit_periodic(a, n)
    # check if index is on the boundary, if yes take value on the opposite for periodic, if not, don't change the value
    a > n ? a = a-n : a < 1 ? a = a+n : a = a
end

@inline function column_to_row(array)
    permutedims(array, reverse(1:ndims(array)))
end