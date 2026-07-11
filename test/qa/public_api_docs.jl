const ROOT_DOCS_SRC = normpath(joinpath(@__DIR__, "..", "..", "docs", "src"))

function _without_julia_comment(line)
    io = IOBuffer()
    in_string = false
    quote_char = '\0'
    escaped = false
    for char in line
        if escaped
            print(io, char)
            escaped = false
        elseif char == '\\'
            print(io, char)
            escaped = true
        elseif in_string
            print(io, char)
            in_string = char != quote_char
        elseif char == '"' || char == '\''
            print(io, char)
            in_string = true
            quote_char = char
        elseif char == '#'
            break
        else
            print(io, char)
        end
    end
    return String(take!(io))
end

function _declared_public_names(statement)
    statement = replace(statement, r"^\s*(export|public|@public|SciMLPublic\.@public)\s+" => "")
    names = Symbol[]
    for part in split(statement, ',')
        token = replace(strip(part), r"\s+" => "")
        occursin(r"^@?[A-Za-z_][A-Za-z0-9_!]*$", token) || continue
        push!(names, Symbol(token))
    end
    return names
end

function _local_public_names(pkg)
    src = joinpath(pkgdir(pkg), "src")
    names = Symbol[]
    isdir(src) || return names
    for (dir, _, filenames) in walkdir(src)
        for filename in filenames
            endswith(filename, ".jl") || continue
            lines = readlines(joinpath(dir, filename))
            in_docstring = false
            i = 1
            while i <= length(lines)
                line = lines[i]
                quote_count = count("\"\"\"", line)
                if quote_count == 1
                    in_docstring = !in_docstring
                elseif quote_count > 1 && isodd(quote_count)
                    in_docstring = !in_docstring
                end
                raw = in_docstring ? "" : _without_julia_comment(line)
                if occursin(r"^\s*(export|public|@public|SciMLPublic\.@public)\b", raw)
                    statement = raw
                    while endswith(rstrip(statement), ",") && i < length(lines)
                        i += 1
                        statement *= " " * _without_julia_comment(lines[i])
                    end
                    append!(names, _declared_public_names(statement))
                end
                i += 1
            end
        end
    end
    return unique(names)
end

function public_api_docs_kwargs(pkg)
    local_names = _local_public_names(pkg)
    reexported_names = setdiff(SciMLTesting.public_api_names(pkg), local_names)
    return (;
        rendered = true,
        docs_src = ROOT_DOCS_SRC,
        ignore = reexported_names,
        rendered_ignore = reexported_names,
    )
end
