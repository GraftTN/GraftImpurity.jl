module GraftImpuritySCSExt

import GraftImpurity
import SCS

function GraftImpurity._pes_conic_backend(::Val{:scs})
    return (;
        optimizer=SCS.Optimizer,
        label="SCS",
        attributes=(
            "eps_abs" => 1e-8,
            "eps_rel" => 1e-8,
            "max_iters" => 100_000,
        ),
    )
end

end
