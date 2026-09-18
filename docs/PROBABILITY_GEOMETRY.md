# Probability geometry — dice, gems, and physical QR

Parent: [Maybe](../README.md) · Theory: [Buleyean probability](./BULEYEAN_PROBABILITY.md) · Geometry: [Index](./INDEX.md)

A probability geek's lark, made concrete: if a weight were a physical object, what
shape would it be? The short answer is that a probability vector has three faithful
physical readings — a solid-angle die, a point on the Bhattacharyya sphere, and an
exact integer glyph — and the repo already owns the shape primitives to render
them. Nothing below claims a physical device works; the executable parts are
geometry and encoding.

## 1. Why a probability is a shape

A convex polyhedron tumbling with uniformly random orientation lands on facet F
with probability equal to the solid angle of F's normal cone divided by 4*pi (the
Gauss map). So a probability vector is a **solid-angle sequence**. A fair d6 is
the regular case: six equal solid angles. A biased die is a non-regular normal
fan.

**Honest boundary:** whether every probability vector is the solid-angle sequence
of some convex polyhedron is a Gauss-map existence question (regularity of the
spherical subdivision), and it is NOT settled here. The lune fan below always
exists, but it is a spherical object, not a tumbling solid.

## 2. The die that always exists: the lune fan

Partition the unit sphere into n lunes. A lune of longitude width D has area 2D,
so lunes of width 2*pi*p_i have areas exactly 4*pi*p_i. This partition always
exists, so every probability vector is realizable as a **direction-space die** (a
spinner on the sphere). Implemented as `luneFan` in `src/probability-die.ts`.

## 3. The gem: the Bhattacharyya sphere

The Bhattacharyya embedding xi_i = 2*sqrt(p_i) lands every distribution on the
sphere of radius 2 (the Fisher-Rao isometry, already in `src/manifold.ts`). A gem
is a point and a direction on that sphere; the geodesic between two distributions
is a great-circle arc.

A sharp consequence: the antipode -xi is NOT a distribution. The complement
distribution is not the antipode, and the sphere's negative orthant is
unreachable from the simplex. The void has no physical point.

## 4. The exact QR

The Buleyean integer weights w_i = n_i + 1 are exact. `encodeWeightQr` packs them
losslessly (base 36, BigInt) and `decodeWeightQr` recovers them; `qrMatrix` renders
a deterministic QR-like glyph (finder corners plus a payload digest) that a camera
could read. For a tamper digest, defer to bitwise `cache-fp48` / `cache-fp64`
rather than inventing a second fingerprint.

## 5. The 55D shapes, physically

The Tensor Bayes state has 55 axes. Only projections are physical, and the repo
already proves the projections: the star/block factorization (`TensorBayesBlockMatrix`),
the 55-axis triangle and its diagonals (`TensorBayesDiagonalSchedule`), and the
Weyl/Conway ladder (`LatticeLadder`).

The bitwise arena supplies the render keys:

- `open-source/bitwise/probability-die.ts` — `probabilityToHelix55` addresses a
  55-outcome distribution by the weighted circular mean on Z_55, using the proven
  period-55 rotation dictionary; `phaseRotation` / `advancePhase` move along the
  helix; `tritDie` / `tritDieDigest` treat a three-outcome distribution as the
  triangular prism and mix its trit carry-free.
- `open-source/bitwise/helix55-rotation-dict.ts` — the 54-strand + parity helix.
- `open-source/bitwise/src/fano-route.ts` — the Fano/Grassmannian mesh ABI, whose
  theorem is `Gnosis.FanoGrassmannianMesh.fanoWirePackedRouteResponse_decode_round_trip`.
- `open-source/bitwise/trit-prism.ts` — the minted balanced-ternary prism.

## 6. What can be done physically (larks)

1. **3D print a die from a weight vector.** monster-studio already turns any file
   into a PLY (`apps/monster-studio/src/lib/local-tools.ts`), so the pipeline is
   weight vector -> lune/Voronoi spec -> PLY -> print. The lune fan is a direction
   die; a Voronoi tiling of a disc is the printable 2D sibling.
2. **The Platonic coincidence.** There are five Platonic solids and five Hope Jar
   rungs (E6/E7/E8/Leech/Kaiju). This is a resonance, not a theorem; do not put it
   on a slide as an identity.
3. **A probability QR as a Voronoi tiling.** Cell areas encode the weights; a
   scanner recovers areas, then the exact integer payload from the glyph digest.
4. **A helix-55 coil.** A physical spring with 55 indexed positions is the
   period-55 rotation dictionary made touchable.

## 7. Reproduce

    a0 run @a0n/maybe:test                      # 39/39 files, includes probability-die.test.ts
    cd open-source/bitwise && ../gnosis/bin/monster test __tests__/probability-die.test.ts

## 8. Honest boundaries

* The die theorem is a uniform-orientation model; real dice are physics.
* The lune fan is spherical, not a convex solid; the polyhedral lift is open.
* `qrMatrix` is a deterministic glyph, not an ISO/IEC 18004 symbol.
* 55D is a projection story; no literal 55-dimensional physical object is implied.
* The complement distribution is not the antipode; the sphere's negative orthant
  is the void, and it is unreachable.

## 9. The Gauss-map existence question (partial answer)

The real question: does every probability vector have a convex die?

**Reduction.** The landing probabilities are the areas of the spherical Voronoi
cells of the facet normals. So the question is an area-constrained spherical
Voronoi problem.

**The equatorial family, exactly solved.** Put every normal on a common latitude.
The bisector of two same-latitude points is a meridian, so each cell is a full
lune of area g_{i-1} + g_i, where g_i is the longitude gap. Requiring
area_i = 4*pi*p_i gives the cyclic system g_{i-1} + g_i = 4*pi*p_i with sum g =
2*pi. It has a unique solution for odd n; for even n it has a solution only when
the alternating condition holds:

    p_0 + p_2 + ... = p_1 + p_3 + ...

So an even-outcome die cannot always be built by putting every normal on one
circle. `analyzeEquatorialVoronoi` solves the system and
`equatorialAlternatingCondition` tests the criterion: (0.3, 0.2, 0.2, 0.3) is
equatorial, (0.4, 0.1, 0.4, 0.1) is not.

**General n.** Surjectivity of the spherical-Voronoi area map is the
semi-discrete optimal-transport result of Aurenhammer-Hoffmann-Aronov and
Kitagawa-Merigot-Thibert (CITED, not proved here). If it holds, every probability
vector is the landing distribution of some convex polyhedron and the die exists.
The equatorial obstruction shows the proof cannot be a single common circle; it
needs the full area map.

So: a direction-space die always exists (section 2); a convex die exists for the
equatorial family exactly as above; for general n it rests on a cited
surjectivity theorem, not a proof in this repo.
