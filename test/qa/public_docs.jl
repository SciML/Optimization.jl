using Test

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

const EXTERNAL_SOURCE_DOCSTRING_EXPORTS = Set(
    [
        "allowsbounds",
        "requiresbounds",
        "allowsconstraints",
        "requiresconstraints",
        "allowscallback",
        "requiresgradient",
        "requireshessian",
        "requiresconsjac",
        "requiresconshess",
    ]
)

function strip_julia_comment(line)
    io = IOBuffer()
    in_string = false
    quote_char = '\0'
    escaped = false
    for c in line
        if escaped
            print(io, c)
            escaped = false
        elseif c == '\\'
            print(io, c)
            escaped = true
        elseif in_string
            print(io, c)
            if c == quote_char
                in_string = false
            end
        elseif c == '"' || c == '\''
            print(io, c)
            in_string = true
            quote_char = c
        elseif c == '#'
            break
        else
            print(io, c)
        end
    end
    return String(take!(io))
end

function public_names_from_statement(statement)
    statement = replace(statement, r"^\s*(export|public|@public|SciMLPublic\.@public)\s+" => "")
    names = String[]
    for part in split(statement, ',')
        token = strip(part)
        isempty(token) && continue
        token = replace(token, r"\s+" => "")
        occursin("...", token) && continue
        occursin(r"^[A-Za-z_][A-Za-z0-9_!]*$", token) || continue
        push!(names, token)
    end
    return names
end

function public_names_from_source(path)
    names = String[]
    lines = readlines(path)
    in_docstring = false
    i = 1
    while i <= length(lines)
        line = lines[i]
        if occursin("\"\"\"", line)
            in_docstring = !in_docstring
            if count("\"\"\"", line) >= 2
                in_docstring = !in_docstring
            end
        end
        raw = in_docstring ? "" : strip_julia_comment(line)
        if occursin(r"^\s*(export|public|@public|SciMLPublic\.@public)\b", raw)
            statement = raw
            while endswith(rstrip(statement), ",") && i < length(lines)
                i += 1
                statement *= " " * strip_julia_comment(lines[i])
            end
            append!(names, public_names_from_statement(statement))
        end
        i += 1
    end
    return names
end

function source_files()
    roots = [joinpath(REPO_ROOT, "src")]
    lib_root = joinpath(REPO_ROOT, "lib")
    append!(roots, [joinpath(lib_root, pkg, "src") for pkg in readdir(lib_root)])
    files = String[]
    for root in roots
        isdir(root) || continue
        for (dir, _, filenames) in walkdir(root)
            append!(
                files,
                joinpath.(Ref(dir), filter(filename -> endswith(filename, ".jl"), filenames))
            )
        end
    end
    return sort(files)
end

function public_names()
    names = String[]
    for file in source_files()
        append!(names, public_names_from_source(file))
    end
    return sort(unique(names))
end

function documented_source_names(path)
    names = String[]
    lines = readlines(path)
    i = 1
    while i <= length(lines)
        if startswith(strip(lines[i]), "@doc \"\"\"")
            i += 1
            while i <= length(lines)
                stripped = strip(lines[i])
                if startswith(stripped, "\"\"\"")
                    target = strip(replace(stripped, r"^\"\"\"\s*" => ""))
                    if isempty(target) && i < length(lines)
                        i += 1
                        target = strip(strip_julia_comment(lines[i]))
                    end
                    m = match(r"([A-Za-z_][A-Za-z0-9_!]*)$", target)
                    m !== nothing && push!(names, m.captures[1])
                    break
                end
                i += 1
            end
        elseif strip(lines[i]) == "\"\"\""
            i += 1
            while i <= length(lines) && strip(lines[i]) != "\"\"\""
                i += 1
            end
            i += 1
            while i <= length(lines) && isempty(strip(lines[i]))
                i += 1
            end
            i > length(lines) && continue
            def = strip(strip_julia_comment(lines[i]))
            def = replace(def, r"^(Base\.)?@kwdef\s+" => "")
            m = match(
                r"^(?:abstract\s+type|mutable\s+struct|struct|primitive\s+type|const|function)\s+([A-Za-z_][A-Za-z0-9_!]*)|^([A-Za-z_][A-Za-z0-9_!]*)\s*\(",
                def,
            )
            if m !== nothing
                push!(names, something(m.captures[1], m.captures[2]))
            end
        else
            i += 1
        end
    end
    return names
end

function documented_source_names()
    names = String[]
    for file in source_files()
        append!(names, documented_source_names(file))
    end
    return Set(names)
end

function docs_entry_names()
    docs_root = joinpath(REPO_ROOT, "docs", "src")
    names = String[]
    for (dir, _, filenames) in walkdir(docs_root)
        for filename in filenames
            endswith(filename, ".md") || continue
            in_docs_block = false
            for line in readlines(joinpath(dir, filename))
                stripped = strip(line)
                if startswith(stripped, "```@docs")
                    in_docs_block = true
                    continue
                elseif in_docs_block && startswith(stripped, "```")
                    in_docs_block = false
                    continue
                end
                in_docs_block || continue
                isempty(stripped) && continue
                entry = first(split(stripped, '('))
                push!(names, last(split(entry, '.')))
            end
        end
    end
    return Set(names)
end

@testset "public API documentation coverage" begin
    names = public_names()
    source_doc_names = documented_source_names()
    docs_names = docs_entry_names()

    missing_source_docstrings = setdiff(
        names,
        union(source_doc_names, EXTERNAL_SOURCE_DOCSTRING_EXPORTS),
    )
    missing_docs_entries = setdiff(names, docs_names)

    @test isempty(missing_source_docstrings)
    @test isempty(missing_docs_entries)
end
