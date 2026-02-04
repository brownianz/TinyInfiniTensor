#include "core/graph.h"
#include "operators/matmul.h"
#include "operators/transpose.h"
#include <algorithm>
#include <numeric>
#include <queue>

namespace infini
{

    void GraphObj::addOperatorAndConnect(const Operator &op)
    {
        sorted = false;
        ops.push_back(op);
        for (auto &input : op->getInputs())
        {
            if (input)
            {
                input->addTarget(op);
                if (auto pred = input->getSource())
                {
                    pred->addSuccessors(op);
                    op->addPredecessors(pred);
                }
            }
        }
        for (auto &output : op->getOutputs())
        {
            if (output)
            {
                output->setSource(op);
                for (auto &succ : output->getTargets())
                {
                    succ->addPredecessors(op);
                    op->addSuccessors(succ);
                }
            }
        }
    }

    string GraphObj::toString() const
    {
        std::ostringstream oss;
        oss << "Graph Tensors:\n";
        for (const auto &tensor : tensors)
            oss << tensor << "\n";

        oss << "Graph operators:\n";
        for (const auto &op : ops)
        {
            vector<UidBaseType> preds, succs;
            for (auto &o : op->getPredecessors())
                preds.emplace_back(o->getGuid());
            for (auto &o : op->getSuccessors())
                succs.emplace_back(o->getGuid());
            oss << "OP " << op->getGuid();
            oss << ", pred " << vecToString(preds);
            oss << ", succ " << vecToString(succs);
            oss << ", " << op << "\n";
        }
        return oss.str();
    }

    bool GraphObj::topo_sort()
    {
        if (this->sorted)
        {
            return true;
        }
        std::vector<Operator> sorted;
        std::unordered_set<OperatorObj *> flags;
        sorted.reserve(ops.size());
        flags.reserve(ops.size());
        while (sorted.size() < ops.size())
        {
            // Any node is move to sorted in this loop.
            auto modified = false;
            for (auto const &op : ops)
            {
                if (auto const &inputs = op->getInputs();
                    flags.find(op.get()) == flags.end() &&
                    std::all_of(inputs.begin(), inputs.end(),
                                [&flags](auto const &input)
                                {
                                    auto ptr = input->getSource().get();
                                    return !ptr || flags.find(ptr) != flags.end();
                                }))
                {
                    modified = true;
                    sorted.emplace_back(op);
                    flags.insert(op.get());
                }
            }
            if (!modified)
            {
                return false;
            }
        }
        this->ops = std::move(sorted);
        return this->sorted = true;
    }

    void GraphObj::optimize()
    {
        // =================================== 作业 ===================================
        // 目标（对齐 test_graph.cc）：
        // 1) 消除相邻的 transpose({0,1,3,2}) + transpose({0,1,3,2})（互相抵消）
        // 2) 将 matmul 输入侧的 transpose({0,1,3,2}) 融合进 matmul 的 transA/transB
        // =================================== 作业 ===================================

        IT_ASSERT(topo_sort() == true);

        const Shape kSwapLast2{0, 1, 3, 2};

        std::unordered_set<OperatorObj *> removed_ops;
        std::unordered_set<TensorObj *> removed_tensors;

        // -------- 1) 消除 transpose + transpose（相邻、同 perm）--------
        for (auto &op : ops)
        {
            if (op->getOpType().underlying() != 2) // transpose
                continue;

            auto trans2 = as<TransposeObj>(op);
            auto in2 = trans2->getInputs()[0];
            auto pred = in2->getSource();
            if (!pred)
                continue;
            if (pred->getOpType().underlying() != 2)
                continue;

            auto trans1 = as<TransposeObj>(pred);

            if (trans1->getPermute() != kSwapLast2 || trans2->getPermute() != kSwapLast2)
                continue;

            auto orig = trans1->getInputs()[0];
            auto out2 = trans2->getOutputs()[0];

            // 将 out2 的所有使用者改为使用 orig
            auto targets = out2->getTargets(); // 拷贝，避免遍历时被修改
            for (auto &user : targets)
            {
                user->replaceInput(out2, orig);
            }

            // 标记删除：两个 transpose op + 中间 tensor(in2) + out2
            removed_ops.insert(trans1.get());
            removed_ops.insert(trans2.get());
            removed_tensors.insert(in2.get());
            removed_tensors.insert(out2.get());
        }

        // -------- 2) 融合 transpose 到 matmul.transA/transB --------
        for (auto &op : ops)
        {
            if (op->getOpType().underlying() != 7) // matmul
                continue;

            auto mm = as<MatmulObj>(op);

            // A(0), B(1)
            for (int idx = 0; idx < 2; ++idx)
            {
                auto in = mm->getInputs()[idx];
                auto src = in->getSource();
                if (!src)
                    continue;

                if (src->getOpType().underlying() != 2)
                    continue;

                auto trans = as<TransposeObj>(src);
                if (trans->getPermute() != kSwapLast2)
                    continue;

                auto orig = trans->getInputs()[0];

                // 用 orig 替换 matmul 的该输入
                mm->replaceInput(in, orig);

                // 融合进 transA/transB（这里用 toggle 更稳）
                if (idx == 0)
                    mm->setTransA(!mm->getTransA());
                else
                    mm->setTransB(!mm->getTransB());

                // 删除该 transpose 以及它的输出 tensor（即 in）
                removed_ops.insert(trans.get());
                removed_tensors.insert(in.get());
            }
        }

        // -------- 3) 从图中移除被标记的 op / tensor --------
        ops.erase(std::remove_if(ops.begin(), ops.end(),
                                 [&](const Operator &op)
                                 { return removed_ops.count(op.get()); }),
                  ops.end());

        tensors.erase(std::remove_if(tensors.begin(), tensors.end(),
                                     [&](const Tensor &t)
                                     { return removed_tensors.count(t.get()); }),
                      tensors.end());

        // 重新标记拓扑序无效（因为图结构变了）
        sorted = false;
        IT_ASSERT(topo_sort() == true);
        // =================================== 作业结束 ===================================
    }

    Tensor GraphObj::getTensor(int fuid) const
    {
        for (auto tensor : tensors)
        {
            if (tensor->getFuid() == fuid)
            {
                return tensor;
            }
        }
        return nullptr;
    }

    void GraphObj::shape_infer()
    {
        for (auto &op : ops)
        {
            auto ans = op->inferShape();
            IT_ASSERT(ans.has_value());
            auto oldOutputs = op->getOutputs();
            IT_ASSERT(ans.value().size() == oldOutputs.size());
            // replace the old outputshape and size with new one
            for (int i = 0; i < (int)ans.value().size(); ++i)
            {
                auto newShape = ans.value()[i];
                auto oldShape = oldOutputs[i]->getDims();
                auto fuid = oldOutputs[i]->getFuid();
                if (newShape != oldShape)
                {
                    auto tensor = this->getTensor(fuid);
                    tensor->setShape(newShape);
                }
            }
        }
    }

    void GraphObj::dataMalloc()
    {
        // topological sorting first
        IT_ASSERT(topo_sort() == true);

        // =================================== 作业 ===================================
        // TODO：利用 allocator 给计算图分配内存
        // HINT: 获取分配好的内存指针后，可以调用 tensor 的 setDataBlob 函数给 tensor 绑定内存
        // =================================== 作业 ===================================

        allocator.info();
    }

    Tensor GraphObj::addTensor(Shape dim, DataType dtype)
    {
        return tensors.emplace_back(make_ref<TensorObj>(dim, dtype, runtime));
    }

    Tensor GraphObj::addTensor(const Tensor &tensor)
    {
        IT_ASSERT(tensor->getRuntime() == runtime,
                  std::string("Tensor runtime mismatch: cannot add a tenosr in ") +
                      tensor->getRuntime()->toString() + " to " +
                      runtime->toString());
        tensors.emplace_back(tensor);
        return tensor;
    }

    TensorVec GraphObj::addTensor(const TensorVec &tensors)
    {
        for (auto &t : tensors)
            addTensor(t);
        return tensors;
    }

    // tensor's "source" and "target" must be in "ops".
    // tensor has no "source" and no "target" must not exist.
    // "inputs" or "outputs" of operators must be in "tensors"
    // "predecessors" and "successors" of an operator of "ops" must be in "ops".
    bool GraphObj::checkValid() const
    {
        for (auto tensor : tensors)
        {
            IT_ASSERT(!(tensor->getTargets().size() == 0 &&
                        nullptr == tensor->getSource()));
            for (auto op : tensor->getTargets())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), op) != ops.end());
            }
            auto op = tensor->getSource();
            IT_ASSERT(!(op && std::find(ops.begin(), ops.end(), op) == ops.end()));
        }
        for (auto op : ops)
        {
            for (auto tensor : op->getInputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto tensor : op->getOutputs())
            {
                IT_ASSERT(std::find(tensors.begin(), tensors.end(), tensor) !=
                          tensors.end());
            }
            for (auto pre : op->getPredecessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), pre) != ops.end());
            }
            for (auto suc : op->getSuccessors())
            {
                IT_ASSERT(std::find(ops.begin(), ops.end(), suc) != ops.end());
            }
        }
        std::set<UidBaseType> s;
        // check whether two tensors with the same FUID exist
        for (auto tensor : tensors)
        {
            int cnt = s.count(tensor->getFuid());
            IT_ASSERT(cnt == 0, std::to_string(tensor->getFuid()));
            s.insert(tensor->getFuid());
        }
        return true;
    }

} // namespace infini
