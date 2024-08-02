# Writing DataFlow Analyses in MLIR

Writing dataflow analyses in MLIR, or well any compiler, can often seem quite
daunting and/or complex. A dataflow analysis generally involves propagating
information about the IR across various different types of control flow
constructs, of which MLIR has many (Block-based branches, Region-based branches,
CallGraph, etc), and it isn't always clear how best to go about performing the
propagation. Dataflow analyses often require implementing fixed-point iteration
when data dependencies form cycles, as can happen with control-flow. Tracking
dependencies and making sure updates are properly propagated can get quite
difficult when writing complex analyses. That is why MLIR provides a framework
for writing general dataflow analyses as well as several utilities to streamline
the implementation of common analyses.

## DataFlow Analysis Framework

MLIR provides a general dataflow analysis framework for building fixed-point
iteration dataflow analyses with ease and utilities for common dataflow
analyses. Because the landscape of IRs in MLIR can be vast, the framework is
designed to be extensible and composable, so that utilities can be shared across
dialects with different semantics as much as possible. The framework also tries
to make debugging dataflow analyses easy by providing (hopefully) insightful
logs with `-debug-only="dataflow"`.

Suppose we want to compute at compile-time the constant-valued results of
operations. For example, consider:

```
%0 = string.constant "foo"
%1 = string.constant "bar"
%2 = string.concat %0, %1
```

We can determine with the information in the IR at compile time the value of
`%2` to be "foobar". This is called constant propagation. In MLIR's dataflow
analysis framework, this is in general called the "analysis state of a program
point"; the "state" being, in this case, the constant value, and the "program
point" being the SSA value `%2`.

The constant value state of an SSA value is implemented as a subclass of
`AnalysisState`, and program points are represented by the `ProgramPoint` union,
which can be operations, SSA values, or blocks. They can also be just about
anything, see [Extending ProgramPoint](#extending-programpoint). In general, an
analysis state represents information about the IR computed by an analysis. The
framework understands states as being either initialized or uninitialized. It
requires the state to implement two functions:

```c++
class AnalysisState {
public:
  /// Indicate whether the analysis state is uninitialized. An analysis state
  /// can be uninitialized on construction and remain uninitialized if no
  /// analysis has provided a value for it. When an analysis state becomes
  /// initialized, it cannot return to be uninitialized.
  virtual bool isUninitialized() const = 0;

  /// Set the analysis state to some initialized value. The dataflow analysis
  /// framework will call this function when the fixed-point iteration has
  /// stalled but uninitialized states remain in-flight. This function generally
  /// sets the analysis state to some pessimistic or "overdefined" value, which
  /// indicates that no useful information about the IR could be determined.
  ///
  /// The function should return `ChangeResult::Change` to indicate that the
  /// value of the state has changed or `ChangeResult::NoChange` to indicate
  /// that calling this function did not modify the value.
  virtual ChangeResult defaultInitialize() = 0;
};
```

Let us define an analysis state to represent a compile time known string value
of an SSA value:

```c++
class StringConstant : AnalysisState {
  /// This is the known string constant value of an SSA value at compile time
  /// as determined by a dataflow analysis. To implement the concept of being
  /// "uninitialized", the potential string value is wrapped in an `Optional`
  /// and set to `None` by default to indicate that no value has been provided.
  Optional<std::string> stringValue = llvm::None;

public:
  /// Return true if no value has been provided for the string constant value.
  bool isUninitialized() const override { return !stringValue.hasValue(); }

  /// Default initialized the state to an empty string. Return whether the value
  /// of the state has changed.
  ChangeResult defaultInitialize() override {
    // If the state already has a value, do nothing.
    if (!isUninitialized())
      return ChangeResult::NoChange;
    // Initialize the state and indicate that its value changed.
    stringValue = "";
    return ChangeResult::Change;
  }

  /// Get the currently known string value.
  StringRef getStringValue() const {
    assert(!isUninitialized() && "getting the value of an uninitialized state");
    return stringValue.getValue();
  }

  /// "Join" the value of the state with another constant.
  ChangeResult join(const Twine &value) {
    // If the current state is uninitialized, just take the value.
    if (isUninitialized()) {
      stringValue = value.str();
      return ChangeResult::Change;
    }
    // If the current state is "overdefined", no new information can be taken.
    if (stringValue->empty())
      return ChangeResult::NoChange;
    // If the current state has a different value, it now has two conflicting
    // values and should go to overdefined.
    if (stringValue != value) {
      stringValue = "";
      return ChangeResult::Change;
    }
    return ChangeResult::NoChange;
  }
};
```

Analysis states often depend on each other. In our example, the constant value
of `%2` depends on that of `%0` and `%1`. It stands to reason that the constant
value of `%2` needs to be recomputed when that of `%0` and `%1` change. The
`DataFlowSolver` implements the fixed-point iteration algorithm and manages the
dependency graph between analysis states.

The computation of analysis states, on the other hand, is performed by dataflow
analyses, subclasses of `DataFlowAnalysis`. A dataflow analysis has to implement
a "transfer function", that is, code that computes the values of some states
using the values of others, and set up the dependency graph correctly. Since the
dependency graph inside the solver is initially empty, it must also set up the
dependency graph.

```c++
class DataFlowAnalysis {
public:
  /// "Visit" the provided program point. This method is typically used to
  /// implement transfer functions on or across program points.
  virtual LogicalResult visit(ProgramPoint point) = 0;

  /// Initialize the dependency graph required by this analysis from the given
  /// top-level operation. This function is called once by the solver before
  /// running the fixed-point iteration algorithm.
  virtual LogicalResult initialize(Operation *top) = 0;
};
```

Dependency management is a little unusual in this framework. The dependents of
the value of a state are not other states but invocations of dataflow analyses
on certain program points. For example:

```c++
class DataFlowSolver {
public:
  /// Add a dependency from an analysis state to a dataflow analysis and a
  /// program point. When `state` is updated, either by getting
  /// default-initialized by the solver or by a dataflow analysis, the solver
  /// will call `analysis->visit(point)`.
  void addDependency(AnalysisState *state, DataFlowAnalysis *analysis,
                     ProgramPoint point);

  /// Indicate to the solver that the given analysis state may have changed. If
  /// it has changed, the solver enqueues the dependent "child analysis
  /// invocations".
  void propagateIfChanged(AnalysisState *state, ChangeResult changed);

  /// Get the analysis state of type `StateT` attached to the program point of
  /// type `PointT`. If one does not yet exist, create one and return it.
  template <typename StateT, typename PointT>
  StateT *getOrCreate(PointT point);
};

class StringConstantPropagation : public DataFlowAnalysis {
public:
  /// Implement the transfer function for string operations. When visiting a
  /// string operation, this analysis will try to determine compile time values
  /// of the operation's results and set them in `StringConstant` states. This
  /// function is invoked on an operation whenever the states of its operands
  /// are changed.
  LogicalResult visit(ProgramPoint point) override {
    // This function expects only to receive operations.
    auto *op = point.get<Operation *>();

    // Get or create the constant string values of the operands.
    SmallVector<StringConstant *> operandValues;
    for (Value operand : op->getOperands()) {
      auto *value = solver.getOrCreate<StringConstant>(operand);
      // Create a dependency from the state to this analysis. When the string
      // value of one of the operation's operands are updated, invoke the
      // transfer function again.
      solver.addDependency(value, this, op);
      // If the state is uninitialized, bail out and come back later when it is
      // initialized.
      if (value->isUninitialized())
        return success();
    }

    // Try to compute a constant value of the result.
    auto *result = solver.getOrCreate<StringConstant>(op->getResult(0));
    for (auto constant = dyn_cast<string::ConstantOp>(op)) {
      // Just grab and set the constant value of the result of the operation.
      // Propagate an update to the state if it changed.
      solver.propagateIfChanged(result, result->join(constant.getValue()));
    } else if (auto concat = dyn_cast<string::ConcatOp>(op)) {
      StringRef lhs = operandValues[0].getStringValue();
      StringRef rhs = operandValues[1].getStringValue();
      // If either operand is overdefined, the results are overdefined.
      if (lhs.empty() || rhs.empty()) {
        solver.propagateIfChanged(result, result->defaultInitialize());

        // Otherwise, compute the constant value and join it with the result.
      } else {
        solver.propagateIfChanged(result, result->join(lhs + rhs));
      }
    } else {
      // We don't know how to implement the transfer function for this
      // operation. Mark its results as overdefined.
      solver.propagateIfChanged(result, result->defaultInitialize());
    }
    return success();
  }
};
```

In the above example, the `visit` function sets up the dependencies of the
analysis invocation on an operation as the constant values of the operands of
each operation. When the operand states have initialized values but overdefined
values, it sets the state of the result to overdefined. Otherwise, it computes
the state of the result and merges the new information in with `join`.

However, the dependency graph still needs to be initialized before the solver
knows what to call `visit` on. This is done in the `initialize` function:

```c++
LogicalResult StringConstantPropagation::initialize(Operation *top) {
  // Visit every nested string operation and set up its dependencies.
  top->walk([&](Operation *op) {
    for (Value operand : op->getOperands()) {
      auto *state = solver.getOrCreate<StringConstant>(operand);
      solver.addDependency(state, this, op);
    }
  });
  // Now that the dependency graph has been set up, "seed" the evolution of the
  // analysis by marking the constant values of all block arguments as
  // overdefined and the results of (non-constant) operations with no operands.
  auto defaultInitializeAll = [&](ValueRange values) {
    for (Value value : values) {
      auto *state = solver.getOrCreate<StringConstant>(value);
      solver.propagateIfChanged(state, state->defaultInitialize());
    }
  };
  top->walk([&](Operation *op) {
    for (Region &region : op->getRegions())
      for (Block &block : region)
        defaultInitializeAll(block.getArguments());
    if (auto constant = dyn_cast<string::ConstantOp>(op)) {
      auto *result = solver.getOrCreate<StringConstant>(constant.getResult());
      solver.propagateIfChanged(result, result->join(constant.getValue()));
    } else if (op->getNumOperands() == 0) {
      defaultInitializeAll(op->getResults());
    }
  });
  // The dependency graph has been set up and the analysis has been seeded.
  // Finish initialization and let the solver run.
  return success();
}
```

Note that we can remove the call to `addDependency` inside our `visit` function
because the dependencies are set by the initialize function. Dependencies added
inside the `visit` function -- that is, while the solver is running -- are
called "dynamic dependencies". Dependending on the kind of analysis, it may be
more efficient to set some dependencies statically or dynamically.

Another way to improve the efficiency of our analysis is to recognize that this
is a *sparse*, *forward* analysis. It is sparse because the dependencies of an
operation's transfer function are only the states of its operands, meaning that
we can track dependencies through the IR instead of relying on the solver to do
the bookkeeping. It is forward (assuming our IR has SSA dominance) because
information can only be propagated from an SSA value's definition to its users.

That is a lot of code to write, however, so the framework comes with utilities
for implementing conditional sparse and dense dataflow analyses. See
[Sparse Forward DataFlowAnalysis](#sparse-forward-dataflow-analysis).

### Running the Solver

Setting up the dataflow solver is straightforward:

```c++
void MyPass::runOnOperation() {
  Operation *top = getOperation();
  DataFlowSolver solver;
  // Load the analysis.
  solver.load<StringConstantPropagation>();
  // Run the solver!
  if (failed(solver.initializeAndRun(top)))
    return signalPassFailure();
  // Query the results and do something...
  top->walk([&](string::ConcatOp concat) {
    auto *result = solver.lookupState<StringConstant>(concat.getResult());
    // ...
  });
}
```

### Extending ProgramPoint

`ProgramPoint` can be extended to represent just about anything in a program:
control-flow edges or memory addresses. Custom "generic" program points are
implemented as subclasses of `GenericProgramPointBase`, a user of the storage
uniquer API, with a content-key.

Example 1: a control-flow edge between two blocks. Suppose we want to represent
the state of an edge in the control-flow graph, such as its liveness. We can
attach such a state to the custom program point:

```c++
/// This program point represents a control-flow edge between two blocks. The
/// block `from` is a predecessor of `to`.
class CFGEdge : public GenericProgramPointBase<CFGEdge,
                                               std::pair<Block *, Block*>> {
public:
  Block *getFrom() const { return getValue().first; }
  Block *getTo() const { return getValue().second; }
};
```

Example 2: a raw memory address after the execution of an operation. This
program point allows us to attach states to a raw memory address before an
operation after an operation is executed.

```c++
class RawMemoryAddr : public GenericProgramPointBase<
    RawMemoryAddr, std::pair<uintptr_t, Operation *>> { /* ... */ };
```

Instances of program points can be accessed as follows:

```c++
Block *from = /* ... */, *to = /* ... */;
auto *cfgEdge = solver.getProgramPoint<CFGEdge>(from, to);

Operation *op = /* ... */;
auto *addr = solver.getProgramPoint<RawMemoryAddr>(0x3000, op);
```

### Multiple State Providers

The dataflow analysis framework is designed to be composable; analyses can be
opaquely mixed together and depend on each other. Multiple analyses can provide
values and information for the same state. For *optimistic* analyses, however,
this can be tricky: what happens if one analysis sets a state to overdefined
while another analysis can still provide values?

The framework (currently) lacks built-in support for this, so setting up these
kinds of analyses requires more work.

TODO: write the details.

## Sparse Forward DataFlow Analysis

One type of dataflow analysis is a sparse forward propagation analysis. This
type of analysis, as the name may suggest, propagates information forward (e.g.
from definitions to uses). The class `SparseDataFlowAnalysis` implements much of
the analysis logic, including handling control-flow, and abstracts away the
dependency management.

To provide a bit of concrete context, let's go over writing a simple forward
dataflow analysis in MLIR. Let's say for this analysis that we want to propagate
information about a special "metadata" dictionary attribute. The contents of
this attribute are simply a set of metadata that describe a specific value, e.g.
`metadata = { likes_pizza = true }`. We will collect the `metadata` for
operations in the IR and propagate them about.

### Lattices

Before going into how one might setup the analysis itself, it is important to
first introduce the concept of a `Lattice` and how we will use it for the
analysis. A lattice represents all of the possible values or results of the
analysis for a given value. A lattice element holds the set of information
computed by the analysis for a given value, and is what gets propagated across

For our analysis in MLIR, we will need to define a class representing the value
held by an element of the lattice used by our dataflow analysis:

```c++
/// The value of our lattice represents the inner structure of a DictionaryAttr,
/// for the `metadata`.
struct MetadataLatticeValue {
  /// Compute a lattice value from the provided dictionary.
  MetadataLatticeValue(DictionaryAttr attr = {})
      : metadata(attr.begin(), attr.end()) {}

  /// Return a pessimistic value state for our value type using only information
  /// about the state of the provided IR. This is similar to the above method,
  /// but may produce a slightly more refined result. This is okay, as the
  /// information is already encoded as fact in the IR.
  static MetadataLatticeValue getPessimisticValueState(Value value) {
    // Check to see if the parent operation has metadata.
    if (Operation *parentOp = value.getDefiningOp()) {
      if (auto metadata = parentOp->getAttrOfType<DictionaryAttr>("metadata"))

  /// Our value represents the combined metadata, which is originally a
  /// DictionaryAttr, so we use a map.
  DenseMap<StringAttr, Attribute> metadata;
};
```

One interesting thing to note above is that we don't have an explicit method for
the `uninitialized` state. This state is handled by the `Lattice` class, which
manages a lattice value for a given IR entity. A quick overview of this class,
and the API that will be interesting to us while writing our analysis, is shown
below:

```c++
/// This class represents a lattice element holding a specific value of type
/// `ValueT`.
template <typename ValueT>
class Lattice : public AbstractLattice {
public:
  /// Return the value held by this element. This requires that a value is
  /// known, i.e. not `uninitialized`.
  ValueT &getValue();
  const ValueT &getValue() const;

  /// Join the information contained in the 'rhs' element into this
  /// element. Returns if the state of the current element changed.
  ChangeResult join(const Lattice<ValueT> &rhs) override;

  /// Join the information contained in the 'rhs' value into this
  /// lattice. Returns if the state of the current lattice changed.
  ChangeResult join(const ValueT &rhs);

  /// Mark the lattice element as having reached a pessimistic fixpoint. This
  /// means that the lattice may potentially have conflicting value states, and
  /// only the conservatively known value state should be relied on.
  ChangeResult markPessimisticFixPoint() override;
};
```

With our lattice defined, we can now define the transfer function that will
compute and propagate our lattice across the IR.

### SparseDataFlowAnalysis Driver

The `SparseDataFlowAnalysis` class represents the driver of the sparse, forward
dataflow analysis, and performs all of the related analysis computation. When
defining our analysis, we will inherit from this class and implement some of its
hooks. Before that, let's look at a quick overview of this class and some of the
important API for our analysis:

```c++
/// This class represents the main driver of the forward dataflow analysis. It
/// takes as a template parameter the value type of lattice being computed.
template <typename ValueT>
class SparseDataFlowAnalysis : public AbstractSparseDataFlowAnalysis {
public:
  using AbstractSparseDataFlowAnalysis::AbstractSparseDataFlowAnalysis;

  /// Return the lattice element attached to the given value. If a lattice has
  /// not been added for the given value, a new 'uninitialized' value is
  /// inserted and returned.
  Lattice<ValueT> *getLatticeElement(Value value) override;

  /// Mark all of the lattice elements for the given range of Values as having
  /// reached a pessimistic fixpoint.
  void markAllPessimisticFixPoint(
      ArrayRef<Lattice<MetadataLatticeValue> *> values);

protected:
  /// Visit the given operation, and join any necessary analysis state
  /// into the lattice elements for the results and block arguments owned by
  /// this operation using the provided set of operand lattice elements
  /// (all pointer values are guaranteed to be non-null). The lattice element
  /// for a result or block argument value can be obtained, and join'ed into, by
  /// using `getLatticeElement`.
  virtual void visitOperation(
      Operation *op, ArrayRef<const Lattice<ValueT> *> operands,
      ArrayRef<Lattice<ValueT> *> results) = 0;
};
```

NOTE: Some API has been redacted for our example. The `SparseDataFlowAnalysis`
contains various other hooks that allow for injecting custom behavior when
applicable.

The main API that we are responsible for defining is the `visitOperation`
method. This method is responsible for computing new lattice elements for the
results and block arguments owned by the given operation. This is where we will
inject the lattice element computation logic, also known as the transfer
function for the operation, that is specific to our analysis. A simple
implementation for our example is shown below:

```c++
class MetadataAnalysis :
    public SparseDataFlowAnalysis<Lattice<MetadataLatticeValue>> {
public:
  using SparseDataFlowAnalysis::SparseDataFlowAnalysis;

  void visitOperation(
        Operation *op, ArrayRef<const Lattice<MetadataLatticeValue> *> operands,
        ArrayRef<Lattice<MetadataLatticeValue> *results) override {
    DictionaryAttr metadata = op->getAttrOfType<DictionaryAttr>("metadata");

    // If we have no metadata for this operation, we will conservatively mark
    // all of the results as having reached a pessimistic fixpoint.
    if (!metadata)
      return markAllPessimisticFixPoint(results);

    // Otherwise, we will compute a lattice value for the metadata and join it
    // into the current lattice element for all of our results.
    MetadataLatticeValue latticeValue(metadata);
    for (Lattice<MetadataLatticeValue> *resultLattice : results) {
      // We join the result lattices with the lattice value for this operation's
      // metadata. Indicate to the analysis whether the lattice has changed.
      propagateIfChanged(resultLattice, resultLattice->join(latticeValue));
    }
  }
};
```

With that, we have all of the necessary components to compute our analysis.
After the analysis has been computed, we can grab any computed information for
values by using `lookupState` on the solver. Note that `lookup` may return a
null value as the analysis is not guaranteed to visit every possible program
point if, for example, a value is in an unreachable block.

```c++
void MyPass::runOnOperation() {
  DataFlowSolver solver;
  solver.load<MyAnalysis>();
  if (failed(solver.initializeAndRun(getOperation())))
    return signalPassFailure();
  ...
}

void MyPass::useAnalysisOn(MetadataAnalysis &analysis, Value value) {
  auto *latticeElement =
      analysis.lookupState<Lattice<MetadataLatticeValue>>(value);

  // If we don't have an element, the `value` wasn't visited during our analysis
  // meaning that it could be dead. We need to treat this conservatively.
  if (!lattice)
    return;

  // Our lattice element has a value, use it:
  MetadataLatticeValue &value = lattice->getValue();
  ...
}
```