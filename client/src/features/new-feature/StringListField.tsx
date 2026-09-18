import { useFieldArray, type Control, type FieldErrors } from 'react-hook-form';
import type { NewFeatureValues } from './schema';
import { Fieldset, RemoveButton } from './fields';

type ListPath = 'goals' | 'constraints' | 'out_of_scope' | 'stakeholders';

/** A repeatable list of plain strings, which the PRD needs in four different places. */
export function StringListField({
  control,
  name,
  legend,
  itemLabel,
  errors,
  minimum = 0,
}: {
  control: Control<NewFeatureValues>;
  name: ListPath;
  legend: string;
  itemLabel: string;
  errors: FieldErrors<NewFeatureValues>;
  minimum?: number;
}) {
  const { fields, append, remove } = useFieldArray({ control, name });
  const listError = errors[name]?.message ?? errors[name]?.root?.message;

  return (
    <Fieldset legend={legend} error={typeof listError === 'string' ? listError : undefined}>
      <ul className="list-field">
        {fields.map((field, index) => (
          <li key={field.id} className="list-field__row">
            <input
              aria-label={`${itemLabel} ${index + 1}`}
              {...control.register(`${name}.${index}.value` as const)}
            />
            {fields.length > minimum ? (
              <RemoveButton onClick={() => remove(index)} label={`Remove ${itemLabel} ${index + 1}`} />
            ) : null}
          </li>
        ))}
      </ul>
      <button type="button" className="button button--quiet" onClick={() => append({ value: '' })}>
        Add {itemLabel.toLowerCase()}
      </button>
    </Fieldset>
  );
}
